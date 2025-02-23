EXTENDED_LOGGING = False

# set to 0 to deactivate writing to keyboard
# try lower values like 0.002 (fast) first, take higher values like 0.05 in case it fails
WRITE_TO_KEYBOARD_INTERVAL = 0.002

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='Start the realtime Speech-to-Text (STT) test with various configuration options.')

    parser.add_argument('-m', '--model', type=str, # no default='large-v2',
                        help='Path to the STT model or model size. Options include: tiny, tiny.en, base, base.en, small, small.en, medium, medium.en, large-v1, large-v2, or any huggingface CTranslate2 STT model such as deepdml/faster-whisper-large-v3-turbo-ct2. Default is large-v2.')

    parser.add_argument('-r', '--rt-model', '--realtime_model_type', type=str, # no default='tiny',
                        help='Model size for real-time transcription. Options same as --model.  This is used only if real-time transcription is enabled (enable_realtime_transcription). Default is tiny.en.')
    
    parser.add_argument('-l', '--lang', '--language', type=str, # no default='en',
                help='Language code for the STT model to transcribe in a specific language. Leave this empty for auto-detection based on input audio. Default is en. List of supported language codes: https://github.com/openai/whisper/blob/main/whisper/tokenizer.py#L11-L110')
    
    parser.add_argument('-d', '--root', type=str, # no default=None,
                help='Root directory where the Whisper models are downloaded to.')

    if EXTENDED_LOGGING:
        import logging
        logging.basicConfig(level=logging.DEBUG)

    from rich.console import Console
    from rich.live import Live
    from rich.text import Text
    from rich.panel import Panel
    from rich.spinner import Spinner
    from rich.progress import Progress, SpinnerColumn, TextColumn
    console = Console()
    console.print("System initializing, please wait")

    import os
    import sys
    from RealtimeSTT import AudioToTextRecorder
    from colorama import Fore, Style
    import colorama
    import pyautogui

    

    if os.name == "nt" and (3, 8) <= sys.version_info < (3, 99):
        from torchaudio._extension.utils import _init_dll_path
        _init_dll_path()    

    colorama.init()

    # Initialize Rich Console and Live
    live = Live(console=console, refresh_per_second=10, screen=False)
    live.start()

    full_sentences = []
    rich_text_stored = ""
    recorder = None
    displayed_text = ""  # Used for tracking text that was already displayed

    end_of_sentence_detection_pause = 0.45
    unknown_sentence_detection_pause = 0.7
    mid_sentence_detection_pause = 2.0

    def clear_console():
        os.system('clear' if os.name == 'posix' else 'cls')

    prev_text = ""

    def preprocess_text(text):
        # Remove leading whitespaces
        text = text.lstrip()

        #  Remove starting ellipses if present
        if text.startswith("..."):
            text = text[3:]

        # Remove any leading whitespaces again after ellipses removal
        text = text.lstrip()

        # Uppercase the first letter
        if text:
            text = text[0].upper() + text[1:]
        
        return text


    def text_detected(text):
        global prev_text, displayed_text, rich_text_stored

        text = preprocess_text(text)

        sentence_end_marks = ['.', '!', '?', '。'] 
        if text.endswith("..."):
            recorder.post_speech_silence_duration = mid_sentence_detection_pause
        elif text and text[-1] in sentence_end_marks and prev_text and prev_text[-1] in sentence_end_marks:
            recorder.post_speech_silence_duration = end_of_sentence_detection_pause
        else:
            recorder.post_speech_silence_duration = unknown_sentence_detection_pause

        prev_text = text

        # console.log(text[text.rfind(' ')+1:])         Notes: the code likely passes the full sentence through a NN to interpret, therefore can't rely on last word.
        searchTriggerWords(text)
        checkToStop()

        # Build Rich Text with alternating colors
        rich_text = Text()
        for i, sentence in enumerate(full_sentences):
            if i % 2 == 0:
                #rich_text += Text(sentence, style="bold yellow") + Text(" ")
                rich_text += Text(sentence, style="yellow") + Text(" ")
            else:
                rich_text += Text(sentence, style="cyan") + Text(" ")
        
        # If the current text is not a sentence-ending, display it in real-time
        if text:
            rich_text += Text(text, style="bold yellow")

        new_displayed_text = rich_text.plain

        if new_displayed_text != displayed_text:
            displayed_text = new_displayed_text
            panel = Panel(rich_text, title="[bold green]Live Transcription[/bold green]", border_style="bold green")
            live.update(panel)
            rich_text_stored = rich_text

    def process_text(text):
        global recorder, full_sentences, prev_text
        recorder.post_speech_silence_duration = unknown_sentence_detection_pause

        text = preprocess_text(text)
        text = text.rstrip()
        if text.endswith("..."):
            text = text[:-2]
                
        if not text:
            return

        full_sentences.append(text)
        prev_text = ""
        text_detected("")

        if WRITE_TO_KEYBOARD_INTERVAL:
            pyautogui.write(f"{text} ", interval=WRITE_TO_KEYBOARD_INTERVAL)  # Adjust interval as needed

    # Recorder configuration
    recorder_config = {
        'spinner': False,
        'model': 'large-v2', # or large-v2 or deepdml/faster-whisper-large-v3-turbo-ct2 or ...
        'download_root': None, # default download root location. Ex. ~/.cache/huggingface/hub/ in Linux
        # 'input_device_index': 1,
        'realtime_model_type': 'tiny.en', # or small.en or distil-small.en or ...
        'language': 'en',
        'silero_sensitivity': 0.05,
        'webrtc_sensitivity': 3,
        'post_speech_silence_duration': unknown_sentence_detection_pause,
        'min_length_of_recording': 1.1,        
        'min_gap_between_recordings': 0,                
        'enable_realtime_transcription': True,
        'realtime_processing_pause': 0.02,
        'on_realtime_transcription_update': text_detected,
        #'on_realtime_transcription_stabilized': text_detected,
        'silero_deactivity_detection': True,
        'early_transcription_on_silence': 0,
        'beam_size': 5,
        'beam_size_realtime': 3,
        # 'batch_size': 0,
        # 'realtime_batch_size': 0,        
        'no_log_file': True,
        'initial_prompt': (
            "End incomplete sentences with ellipses.\n"
            "Examples:\n"
            "Complete: The sky is blue.\n"
            "Incomplete: When the sky...\n"
            "Complete: She walked home.\n"
            "Incomplete: Because he...\n"
        )
    }

    args = parser.parse_args()
    if args.model is not None:
        recorder_config['model'] = args.model
        print(f"Argument 'model' set to {recorder_config['model']}")
    if args.rt_model is not None:
        recorder_config['realtime_model_type'] = args.rt_model
        print(f"Argument 'realtime_model_type' set to {recorder_config['realtime_model_type']}")
    if args.lang is not None:
        recorder_config['language'] = args.lang
        print(f"Argument 'language' set to {recorder_config['language']}")
    if args.root is not None:
        recorder_config['download_root'] = args.root
        print(f"Argument 'download_root' set to {recorder_config['download_root']}")

    if EXTENDED_LOGGING:
        recorder_config['level'] = logging.DEBUG

    recorder = AudioToTextRecorder(**recorder_config)
    
    initial_text = Panel(Text("Say something...", style="cyan bold"), title="[bold yellow]Waiting for Input[/bold yellow]", border_style="bold yellow")
    live.update(initial_text)

    import string
    from time import sleep
    from datafeel.device import VibrationMode, discover_devices, LedMode, ThermalMode, VibrationWaveforms

    import json

    

    devices = discover_devices(4)
    print(len(devices))
    device = devices[0]

    def loadTriggerWords():
        with open('wordTriggers.json') as f:
            return json.load(f)
        
    wordBankTest = loadTriggerWords()

    
    import time
    running = False
    start_time = 0.0

    def checkToStop():
        global running, start_time
        if running and time.time() - start_time > 1:
            console.log("STOPPING")
            running = False

            for d in devices:
                d.stop_vibration()
                d.set_led(0, 0, 0)
                d.disable_all_thermal()


    tree_colors = [[252, 97, 130],
     [255, 255, 0],
     [139, 69, 19],
     [251, 125, 173],
     [142, 94, 146]]
    
    fruit_colors = [[255, 0, 0],
                    [255, 255, 0],
                    [139, 69, 19],
                    [56, 173, 250],
                    [113, 44, 72]]
    
    ripple_sequence = [
        VibrationWaveforms.STRONG_CLICK1_P100,
        VibrationWaveforms.Rest(0.5),
        VibrationWaveforms.STRONG_CLICK3_P60,
        VibrationWaveforms.Rest(0.5),
        VibrationWaveforms.STRONG_CLICK4_P30,
        VibrationWaveforms.Rest(0.5),
        VibrationWaveforms.END_SEQUENCE,
    ]

    import random
    def searchTriggerWords(text):
        global start_time, running
        trg = text.maketrans("", "", string.punctuation)
        tokens = text.translate(trg).lower().split()
        
        for token in tokens:
            if token in wordBankTest:
                device.registers.set_led_mode(LedMode.GLOBAL_MANUAL)
                device.registers.set_vibration_mode(VibrationMode.LIBRARY)


                console.log(text)
                temperature = wordBankTest[token]['Temperature']
                vibration = wordBankTest[token]['Vibration']
                light = wordBankTest[token]['Light']

                for d in devices:
                    match token:    # Color symbols
                        case 'trees':
                            color_ind = random.randrange(len(tree_colors))
                            d.set_led(tree_colors[color_ind][0], tree_colors[color_ind][1], tree_colors[color_ind][2])
                        case 'fruits':
                            color_ind = random.randrange(len(fruit_colors))
                            d.set_led(fruit_colors[color_ind][0], fruit_colors[color_ind][1], fruit_colors[color_ind][2])
                        case _:
                            d.set_led(light[0], light[1], light[2])
                    
                    match token:    # Vibration
                        case 'splashing':
                            d.play_vibration_sequence(ripple_sequence)
                        case 'rang':
                            d.play_vibration_sequence(ripple_sequence)
                        case 'sneeze':
                            d.play_vibration_sequence([VibrationWaveforms.STRONG_CLICK1_P100,
                                                       VibrationWaveforms.Rest(1),
                                                       VibrationWaveforms.END_SEQUENCE,])
                        case 'humming':
                            d.play_vibration_sequence([VibrationWaveforms.SMOOTH_HUM_P50,
                                                       VibrationWaveforms.Rest(0.5),
                                                       VibrationWaveforms.SMOOTH_HUM_P50,
                                                       VibrationWaveforms.Rest(0.5),
                                                       VibrationWaveforms.SMOOTH_HUM_P50,
                                                       VibrationWaveforms.Rest(0.5),
                                                       VibrationWaveforms.END_SEQUENCE,])
                        case _:
                            d.play_frequency(250, vibration)

                    d.activate_thermal_intensity_control(temperature) # Heating


                # device.play_frequency(250, vibration)
                # device.set_led(light[0], light[1], light[2])
                # device.activate_thermal_intensity_control(temperature) # Heating
                start_time = time.time()
                running = True
                return
    
    

    try:
        while True:
            recorder.text(process_text)
            checkToStop()

    except KeyboardInterrupt:
        live.stop()
        console.print("[bold red]Transcription stopped by user. Exiting...[/bold red]")
        exit(0)
