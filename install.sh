pipenv --python 3.7

pipenv install pandas==1.0.3 \
numpy==1.18.3 \
matplotlib==3.2.1 \
pillow==7.1.2 \
wordcloud==1.6.0

pipenv install black --dev --pre \
mypy==0.770 --dev \
pytest==5.4.1 --dev