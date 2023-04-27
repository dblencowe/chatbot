.create-env:
	python -m venv ./.venv-chatbot

.activate-env:
	. ./.venv-chatbot/bin/activate

.install-requirements: .activate-env
	pip install -r requirements.txt

setup: .create-env

train: .activate-env .install-requirements
	python train.py

chat: .activate-env .install-requirements
	python chat.py