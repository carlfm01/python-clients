# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

import argparse
from pathlib import Path
import json
import grpc
import riva.client
from riva.client.argparse_utils import add_asr_config_argparse_parameters, add_connection_argparse_parameters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline file transcription via Riva AI Services. \"Offline\" means that entire audio "
        "content of `--input-file` is sent in one request and then a transcript for whole file recieved in "
        "one response.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-file", required=True, type=Path, help="A path to a local file to transcribe.")
    parser.add_argument("--output-json", type=Path, help="Path to save the transcription result as a JSON file.")
    parser = add_connection_argparse_parameters(parser)
    parser = add_asr_config_argparse_parameters(parser, max_alternatives=True, profanity_filter=True, word_time_offsets=True)
    args = parser.parse_args()
    args.input_file = args.input_file.expanduser()
    return args

def print_offline_json(response, output_file: Path = None) -> None:
    """
    Print the response and optionally save it to a JSON file.
    """
    # Convert the response to a dictionary
    response_dict = {
        "results": [
            {
                "alternatives": [
                    {
                        "transcript": alt.transcript,
                        "confidence": alt.confidence,
                        "words": [
                            {
                                "word": word.word,
                                "start_time": word.start_time,
                                "end_time": word.end_time,
                                "confidence": word.confidence,
                                "speaker_tag": word.speaker_tag,
                            }
                            for word in alt.words
                        ],
                    }
                    for alt in res.alternatives
                ]
            }
            for res in response.results
        ]
    }

    if output_file:
        with output_file.open("w", encoding="utf-8") as f:
            json.dump(response_dict, f, indent=4, ensure_ascii=False)

def main() -> None:
    args = parse_args()
    auth = riva.client.Auth(args.ssl_cert, args.use_ssl, args.server, args.metadata)
    asr_service = riva.client.ASRService(auth)
    config = riva.client.RecognitionConfig(
        language_code=args.language_code,
        max_alternatives=args.max_alternatives,
        profanity_filter=args.profanity_filter,
        enable_automatic_punctuation=args.automatic_punctuation,
        verbatim_transcripts=not args.no_verbatim_transcripts,
        enable_word_time_offsets=args.word_time_offsets or args.speaker_diarization,
    )
    riva.client.add_word_boosting_to_config(config, args.boosted_lm_words, args.boosted_lm_score)
    riva.client.add_speaker_diarization_to_config(config, args.speaker_diarization, args.diarization_max_speakers)
    riva.client.add_endpoint_parameters_to_config(
        config,
        args.start_history,
        args.start_threshold,
        args.stop_history,
        args.stop_history_eou,
        args.stop_threshold,
        args.stop_threshold_eou
    )
    riva.client.add_custom_configuration_to_config(
        config,
        args.custom_configuration
    )
    with args.input_file.open('rb') as fh:
        data = fh.read()
    try:
        response=asr_service.offline_recognize(data, config)
        print_offline_json(response, args.output_json)
    except grpc.RpcError as e:
        print(e.details())


if __name__ == "__main__":
    main()
