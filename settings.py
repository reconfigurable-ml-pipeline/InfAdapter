import os

from decouple import config


BASE_DIR = os.path.abspath(os.path.dirname(__file__))

NASA_LOGS_DIRECTORY = config("nasa_logs_directory")
if NASA_LOGS_DIRECTORY[-1] == "/":
    NASA_LOGS_DIRECTORY = NASA_LOGS_DIRECTORY[:-1]

WORLDCUP_LOGS_DIRECTORY = config("worldcup_logs_directory")
if WORLDCUP_LOGS_DIRECTORY[-1] == "/":
    WORLDCUP_LOGS_DIRECTORY = WORLDCUP_LOGS_DIRECTORY[:-1]
