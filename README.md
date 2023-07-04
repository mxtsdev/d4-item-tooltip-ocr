# Diablo IV: Item Tooltip OCR

Utilizing PaddleOCR and a custom Diablo 4 trained recognition model. Outputs item tooltip data in JSON-format.


## Installation

- Clone repository

- Create a Python3 enviroment (I recommend using https://www.anaconda.com/download on Windows)

- pip install -r requirements.txt


## Using

### Output json to console
```
python d4-item-tooltip-ocr.py --source-img=examples\screenshot_001.png
python d4-item-tooltip-ocr.py --source-img=examples\tooltip_001.jpg --find-tooltip=False
```

### Output json to file
```
python d4-item-tooltip-ocr.py --source-img=examples\screenshot_001.png --json-output=item-tooltip-data.json
python d4-item-tooltip-ocr.py --source-img=examples\tooltip_001.jpg --json-output=item-tooltip-data.json --find-tooltip=False
```

### Debug mode
```
python d4-item-tooltip-ocr.py --debug=True --source-img=examples\screenshot_001.png
python d4-item-tooltip-ocr.py --debug=True --source-img=examples\tooltip_001.jpg --find-tooltip=False
```