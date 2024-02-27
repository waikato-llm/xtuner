# Copyright (c) OpenMMLab. All rights reserved.
# Copyright (C) University of Waikato, Hamilton, NZ
import argparse
import json
import re
import traceback

import torch
from datetime import datetime
from peft import PeftModel
from transformers import (AutoModel, AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, CLIPImageProcessor,
                          CLIPVisionModel, GenerationConfig)

from xtuner.tools.utils import get_stop_criteria
from xtuner.utils import (DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX,
                          PROMPT_TEMPLATE, SYSTEM_TEMPLATE)
from xtuner.tools.plugins import plugins_api
from rdh import Container, MessageContainer, configure_redis, run_harness, log

TORCH_DTYPE_MAP = dict(
    fp16=torch.float16,
    bf16=torch.bfloat16,
    fp32=torch.float32,
    auto='auto'
)


def remove_prefix(state_dict, prefix):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith(prefix):
            new_key = key[len(prefix):]
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    return new_state_dict


def process_text(msg_cont):
    """
    Processes the message container, loading the text from the message and forwarding the model output.

    :param msg_cont: the message container to process
    :type msg_cont: MessageContainer
    """
    config = msg_cont.params.config
    args = config.args

    try:
        start_time = datetime.now()
        if config.verbose:
            log("process_texts - start processing text")

        # read data
        d = json.loads(msg_cont.message['data'].decode())
        text = d["text"] if ("text" in d) else ""
        inputs = d["history"] if ("history" in d) else ""
        n_turn = d["turns"] if ("turns" in d) else 0

        if args.prompt_template:
            prompt_text = ''
            template = PROMPT_TEMPLATE[args.prompt_template]
            if 'SYSTEM' in template and n_turn == 0:
                system_text = None
                if args.system_template is not None:
                    system_text = SYSTEM_TEMPLATE[
                        args.system_template].format(
                            round=n_turn + 1, bot_name=args.bot_name)
                elif args.system is not None:
                    system_text = args.system
                if system_text is not None:
                    prompt_text += template['SYSTEM'].format(
                        system=system_text,
                        round=n_turn + 1,
                        bot_name=args.bot_name)
            prompt_text += template['INSTRUCTION'].format(
                input=text, round=n_turn + 1, bot_name=args.bot_name)
            if args.prompt_template == args.system_template == 'moss_sft':
                if not config.inner_thoughts_open:
                    prompt_text.replace('- Inner thoughts: enabled.',
                                        '- Inner thoughts: disabled.')
                if not config.calculate_open:
                    prompt_text.replace(('- Calculator: enabled. API: '
                                         'Calculate(expression)'),
                                        '- Calculator: disabled.')
                if not config.solve_open:
                    prompt_text.replace(
                        '- Equation solver: enabled. API: Solve(equation)',
                        '- Equation solver: disabled.')
                if not config.search_open:
                    prompt_text.replace(
                        '- Web search: enabled. API: Search(query)',
                        '- Web search: disabled.')
        else:
            prompt_text = text
        inputs += prompt_text

        if n_turn == 0:
            ids = config.tokenizer.encode(inputs, return_tensors='pt')
        else:
            ids = config.tokenizer.encode(
                inputs, return_tensors='pt', add_special_tokens=False)

        if args.with_plugins is not None:
            generate_output = config.llm.generate(
                inputs=ids.cuda(),
                generation_config=config.gen_config,
                streamer=None,
                stopping_criteria=config.stop_criteria).cpu()
            generate_output_text = config.tokenizer.decode(
                generate_output[0][len(ids[0]):])
            if config.verbose:
                log(generate_output_text)
            pattern = r'<\|Commands\|>:(.*?)<eoc>'
            command_text = ', '.join(
                re.findall(pattern, generate_output_text))
            extent_text = plugins_api(
                command_text,
                calculate_open=config.calculate_open,
                solve_open=config.solve_open,
                search_open=config.search_open)
            if config.verbose:
                log(extent_text)
            extent_text_ids = config.tokenizer.encode(
                extent_text,
                return_tensors='pt',
                add_special_tokens=False)
            new_ids = torch.cat((generate_output, extent_text_ids),
                                dim=1)

            generate_output = config.llm.generate(
                inputs=new_ids.cuda(),
                generation_config=config.gen_config,
                streamer=None,
                stopping_criteria=config.stop_criteria)
            output_text = config.tokenizer.decode(
                generate_output[0][len(new_ids[0]):])
            if config.verbose:
                log(output_text)
        else:
            generate_output = config.llm.generate(
                inputs=ids.cuda(),
                generation_config=config.gen_config,
                streamer=None,
                stopping_criteria=config.stop_criteria)
            output_text = config.tokenizer.decode(
                generate_output[0][len(ids[0]):])
            if config.verbose:
                log(output_text)
        inputs = config.tokenizer.decode(generate_output[0])

        n_turn += 1
        inputs += config.sep
        if len(generate_output[0]) >= args.max_new_tokens:
            log(
                'Remove the memory of history responses, since '
                f'it exceeds the length limitation of {args.max_new_tokens} tokens.')
            n_turn = 0
            inputs = ''

        # send result
        d = dict()
        d["text"] = str(output_text)
        if not args.no_history:
            d["turns"] = n_turn
            d["history"] = inputs
        msg_cont.params.redis.publish(msg_cont.params.channel_out, json.dumps(d))

        if config.verbose:
            log("process_images - prediction image published: %s" % msg_cont.params.channel_out)
            end_time = datetime.now()
            processing_time = end_time - start_time
            processing_time = int(processing_time.total_seconds() * 1000)
            log("process_images - finished processing image: %d ms" % processing_time)

    except KeyboardInterrupt:
        msg_cont.params.stopped = True
    except:
        log("process_images - failed to process: %s" % traceback.format_exc())


def parse_args():
    parser = argparse.ArgumentParser(description='Chat with a HF model')
    parser.add_argument(
        'model_name_or_path', help='Hugging Face model name or path')
    adapter_group = parser.add_mutually_exclusive_group()
    adapter_group.add_argument(
        '--adapter', default=None, help='adapter name or path')
    parser.add_argument(
        '--torch-dtype',
        default='fp16',
        choices=TORCH_DTYPE_MAP.keys(),
        help='Override the default `torch.dtype` and load the model under '
        'a specific `dtype`.')
    parser.add_argument(
        '--prompt-template',
        choices=PROMPT_TEMPLATE.keys(),
        default=None,
        help='Specify a prompt template')
    system_group = parser.add_mutually_exclusive_group()
    system_group.add_argument(
        '--system', default=None, help='Specify the system text')
    system_group.add_argument(
        '--system-template',
        choices=SYSTEM_TEMPLATE.keys(),
        default=None,
        help='Specify a system template')
    parser.add_argument(
        '--bits',
        type=int,
        choices=[4, 8, None],
        default=None,
        help='LLM bits')
    parser.add_argument(
        '--bot-name', type=str, default='BOT', help='Name for Bot')
    parser.add_argument(
        '--with-plugins',
        nargs='+',
        choices=['calculate', 'solve', 'search'],
        help='Specify plugins to use')
    parser.add_argument(
        '--stop-words', nargs='+', type=str, default=[], help='Stop words')
    parser.add_argument(
        '--offload-folder',
        default=None,
        help='The folder in which to offload the model weights (or where the '
        'model weights are already offloaded).')
    parser.add_argument(
        '--max-new-tokens',
        type=int,
        default=2048,
        help='Maximum number of new tokens allowed in generated text')
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.1,
        help='The value used to modulate the next token probabilities.')
    parser.add_argument(
        '--top-k',
        type=int,
        default=40,
        help='The number of highest probability vocabulary tokens to '
        'keep for top-k-filtering.')
    parser.add_argument(
        '--top-p',
        type=float,
        default=0.75,
        help='If set to float < 1, only the smallest set of most probable '
        'tokens with probabilities that add up to top_p or higher are '
        'kept for generation.')
    parser.add_argument(
        '--repetition-penalty',
        type=float,
        default=1.0,
        help='The parameter for repetition penalty. 1.0 means no penalty.')
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='Random seed for reproducible text generation')

    parser.add_argument(
        '--no_history', action='store_true', help='Whether discard and prior inputs or accumulate them')
    parser.add_argument(
        '--verbose', action='store_true', help='Whether to be more verbose with the output')

    # redis parameters
    parser.add_argument('--redis_host', metavar='HOST', required=False, default="localhost", type=str,
                        dest='redis_host', help='The redis server to connect to')
    parser.add_argument('--redis_port', metavar='PORT', required=False, default=6379, type=int,
                        dest='redis_port', help='The port the redis server is listening on')
    parser.add_argument('--redis_password', metavar='PASSWORD', required=False, default=None, type=str,
                        dest='redis_password', help='The password to use for connecting')
    parser.add_argument('--redis_password_env', metavar='PASSWORD', required=False, default=None, type=str,
                        dest='redis_password_env', help='The environment variable to obtain the password from to use for connecting')
    parser.add_argument('--redis_db', metavar='DB', required=False, default=0, type=int,
                        dest='redis_db', help='The redis database to use')
    parser.add_argument('--redis_in', metavar='CHANNEL', required=True, default=None, type=str,
                        dest='redis_in', help='The redis channel to receive the data from')
    parser.add_argument('--redis_out', metavar='CHANNEL', required=True, default=None, type=str,
                        dest='redis_out', help='The redis channel to publish the processed data on')
    parser.add_argument('--redis_timeout', metavar='NUM', required=False, default=0.01, type=float,
                        dest='redis_timeout', help='The timeout to use for the pubsub thread sleep_time parameter.')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    config = Container()
    config.verbose = args.verbose
    config.args = args

    # build llm
    quantization_config = None
    load_in_8bit = False
    if args.bits == 4:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            load_in_8bit=False,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4')
    elif args.bits == 8:
        load_in_8bit = True
    model_kwargs = {
        'quantization_config': quantization_config,
        'load_in_8bit': load_in_8bit,
        'device_map': 'auto',
        'offload_folder': args.offload_folder,
        'trust_remote_code': True,
        'torch_dtype': TORCH_DTYPE_MAP[args.torch_dtype]
    }
    if args.with_plugins is None:
        inner_thoughts_open = False
        calculate_open = False
        solve_open = False
        search_open = False
    else:
        assert args.prompt_template == args.system_template == 'moss_sft'
        inner_thoughts_open = True
        calculate_open = 'calculate' in args.with_plugins
        solve_open = 'solve' in args.with_plugins
        search_open = 'search' in args.with_plugins
        # pre-import for api and model preparation
        if calculate_open:
            from plugins import calculate  # noqa: F401
        if solve_open:
            from plugins import solve  # noqa: F401
        if search_open:
            from plugins import search  # noqa: F401
    # build llm
    log(f'Load LLM: {args.model_name_or_path}')
    llm = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        encode_special_tokens=True)
    if args.adapter is not None:
        llm = PeftModel.from_pretrained(
            llm,
            args.adapter,
            offload_folder=args.offload_folder,
            trust_remote_code=True)
        log(f'Load adapter: {args.adapter}')

    llm.eval()

    stop_words = args.stop_words
    sep = ''
    if args.prompt_template:
        log(f'Prompt template: {args.prompt_template}')
        template = PROMPT_TEMPLATE[args.prompt_template]
        stop_words += template.get('STOP_WORDS', [])
        sep = template.get('SEP', '')
    stop_criteria = get_stop_criteria(
        tokenizer=tokenizer, stop_words=stop_words)

    gen_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        do_sample=args.temperature > 0,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
    )

    config.llm = llm
    config.sep = sep
    config.gen_config = gen_config
    config.tokenizer = tokenizer
    config.stop_criteria = stop_criteria
    config.inner_thoughts_open = inner_thoughts_open
    config.calculate_open = calculate_open
    config.solve_open = solve_open
    config.search_open = search_open

    params = configure_redis(args, config=config)
    run_harness(params, process_text)


if __name__ == '__main__':
    main()
