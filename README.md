# Diablo IV: Item Tooltip OCR

Utilizing PaddleOCR and a custom Diablo 4 trained recognition model. Outputs item tooltip data in JSON-format.


## Example

![item tooltip](https://github.com/mxtsdev/d4-item-tooltip-ocr/assets/58796811/06aa5a7d-23a4-40c3-9d57-1916d3d3d32b)

```json
{
   "affixes": [
      "+22.5% Overpower Damage [22.5]%",
      "+12.5% Damage to Slowed Enemies [11.5 - 16.5]%",
      "+14.0% Critical Strike Damage [10.0 - 15.0]%",
      "+44 Willpower +[41 - 51]",
      "+7.0% Damage Over Time [5.0 - 10.0]%"
   ],
   "aspect": "Core Skills deal an additional 7.0%[x] [6.0 - 8.0]% damage for each active Companion. (Druid Only)",
   "item_power": "710",
   "item_power_upgraded": null,
   "item_upgrades_current": null,
   "item_upgrades_max": null,
   "name": "SHEPHERD'S WOLF'S BITE",
   "sockets": [],
   "stats": [
      "806 Damage Per Second (-1555)",
      "[586 - 880] Damage per Hit",
      "1.10 Attacks per Second (Fast Weapon)"
   ],
   "type": "Sacred Legendary Mace"
}
```


## Installation

- Clone repository

- Create a Python3 enviroment (I recommend using https://www.anaconda.com/download on Windows)

- pip install -r requirements.txt

You will need to install the correct version of PaddlePaddle depending on your environment (CPU/GPU/CUDA version). Please refer to this link:
https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/en/install/pip/windows-pip_en.html#old-version-anchor-3-INSTALLATION

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
