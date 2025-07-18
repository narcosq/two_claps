import io
import time
import webbrowser
import numpy as np
import pyaudio
import pygame
import speech_recognition as sr
from dotenv import load_dotenv
from gtts import gTTS
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from scipy import signal

load_dotenv()


class VoiceActivatedAI:
    def __init__(self):
        self.Fs = 44100
        self.frame_duration = 0.02
        self.frame_size = int(self.Fs * self.frame_duration)
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.f_low = 1400
        self.f_high = 1800
        self.order = 2
        self.sos = signal.butter(
            self.order,
            [self.f_low, self.f_high],
            btype="bandpass",
            fs=self.Fs,
            output="sos",
        )
        self.window = signal.windows.hann(self.frame_size)
        self.threshold = 0.2
        self.min_peak_distance_sec = 0.2
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        pygame.mixer.init()
        self.setup_ai_agent()

    def setup_ai_agent(self):
        @tool
        def open_browser(url: str) -> str:
            """Открывает указанный URL в веб-браузере Firefox."""
            webbrowser.open(url)
            return f"Открыл браузер по адресу {url}"

        self.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
        self.tools = [open_browser]
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Ты — полезный голосовой ассистент. Ты можешь открывать URL в браузере Firefox. "
                    "Отвечай кратко и по делу, так как твои ответы будут преобразованы в речь.",
                ),
                ("human", "{input}"),
                ("placeholder", "{agent_scratchpad}"),
            ]
        )
        agent = create_tool_calling_agent(prompt=self.prompt, tools=self.tools, llm=self.llm)
        self.agent_executor = AgentExecutor(agent=agent, tools=self.tools, verbose=True)

    def detect_claps(self):
        print("Ожидаю хлопков... Пожалуйста, хлопните дважды, чтобы активировать ассистента.")
        self.stream = self.p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.Fs,
            input=True,
            frames_per_buffer=self.frame_size,
        )
        clap_count = 0
        last_peak_time = -float("inf")
        zi = np.zeros((self.sos.shape[0], 2))
        start_time = time.time()
        while clap_count < 2:
            try:
                frame = np.frombuffer(
                    self.stream.read(self.frame_size, exception_on_overflow=False),
                    dtype=np.float32,
                )
                frame = frame * self.window
                frame_filtered, zi = signal.sosfilt(self.sos, frame, zi=zi)
                peaks, _ = signal.find_peaks(np.abs(frame_filtered), height=self.threshold)
                current_time = time.time() - start_time
                if len(peaks) > 0 and (current_time - last_peak_time) >= self.min_peak_distance_sec:
                    clap_count += 1
                    print(f"Обнаружен хлопок: {current_time:.2f} сек. (Счетчик: {clap_count}/2)")
                    last_peak_time = current_time
            except Exception as e:
                print(f"Ошибка при обнаружении хлопков: {e}")
                continue
        self.stream.stop_stream()
        self.stream.close()
        print("Два хлопка обнаружены! Активация ассистента...")
        return True

    def play_intro_sound(self):
        try:
            tts = gTTS(text="Здравствуйте, чем могу помочь?", lang="ru")
            audio_buffer = io.BytesIO()
            tts.write_to_fp(audio_buffer)
            audio_buffer.seek(0)
            pygame.mixer.music.load(audio_buffer)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
        except Exception as e:
            print(f"Ошибка при воспроизведении приветствия: {e}")
            print("Здравствуйте, чем могу помочь?")

    def listen_for_speech(self, timeout=5):
        print(f"Слушаю вашу команду в течение {timeout} секунд...")
        try:
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=timeout)
            text = self.recognizer.recognize_google(audio, language="ru-RU")
            print(f"Вы сказали: {text}")
            return text
        except sr.WaitTimeoutError:
            print("За отведенное время речь не обнаружена.")
            return None
        except sr.UnknownValueError:
            print("Не удалось распознать речь.")
            return None
        except sr.RequestError as e:
            print(f"Ошибка сервиса распознавания речи: {e}")
            return None

    def process_with_ai_agent(self, user_input):
        try:
            print(f"Обрабатываю запрос: {user_input}")
            response = self.agent_executor.invoke(input={"input": user_input})
            output_text = response.get("output", "К сожалению, я не смог обработать ваш запрос.")
            print(f"Ответ ИИ: {output_text}")
            return output_text
        except Exception as e:
            error_message = f"Извините, произошла ошибка при обработке вашего запроса: {str(e)}"
            print(error_message)
            return error_message

    def text_to_speech(self, text):
        try:
            print(f"Преобразую в речь: {text}")
            tts = gTTS(text=text, lang="ru")
            audio_buffer = io.BytesIO()
            tts.write_to_fp(audio_buffer)
            audio_buffer.seek(0)
            pygame.mixer.music.load(audio_buffer)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
        except Exception as e:
            print(f"Ошибка при преобразовании текста в речь: {e}")
            print(f"Ассистент говорит: {text}")

    def run(self):
        try:
            while True:
                if self.detect_claps():
                    self.play_intro_sound()
                    user_speech = self.listen_for_speech(timeout=5)
                    if user_speech:
                        ai_response = self.process_with_ai_agent(user_speech)
                        self.text_to_speech(ai_response)
                    else:
                        self.text_to_speech("Я ничего не расслышала. Пожалуйста, попробуйте снова, хлопнув дважды.")
                    print("\n" + "=" * 50)
                    print("Готов к следующей активации. Хлопните дважды для продолжения...")
                    print("=" * 50 + "\n")
        except KeyboardInterrupt:
            print("\nЗавершение работы голосового ассистента...")
        except Exception as e:
            print(f"Непредвиденная ошибка: {e}")
        finally:
            if self.stream and not self.stream.is_stopped():
                self.stream.stop_stream()
                self.stream.close()
            self.p.terminate()
            pygame.mixer.quit()


def main():
    print("- Хлопните дважды для активации")
    assistant = VoiceActivatedAI()
    assistant.run()


if __name__ == "__main__":
    main()
