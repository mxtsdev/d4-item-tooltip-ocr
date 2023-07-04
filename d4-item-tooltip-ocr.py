import cv2
import numpy as np
import json
import re
import os
import tkinter as tk
import argparse
from paddleocr import PaddleOCR, draw_ocr
from Levenshtein import ratio

class LineEntry():
	def __init__(self, index, txt, boxes, score):
		self.index = index
		self.txt = txt
		self.score = score
		x = [b[0] for b in boxes]
		y = [b[1] for b in boxes]
		self.tl = (min(x), min(y)) #top-left
		self.br = (max(x), max(y)) #bottom-right
		self.dy = self.br[1] - self.tl[1]
		self.cy = self.tl[1] + int(self.dy / 2.0)

	@staticmethod
	def EMPTY():
		return LineEntry(-1, None, [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], 0.0)

	def re_get(self, pattern: str, *return_group_names, flags = 0):
		if self.index == -1: return (*[None for gn in return_group_names],)

		m = re.search(pattern, self.txt)
		results = [m.group(gn) for gn in return_group_names]
		return (*results,)

class LineCollection():
	def __init__(self, result):
		if isinstance(result, list) and all(isinstance(x, LineEntry) for x in result):
			self.lines = result
		elif isinstance(result, list) and len(result) == 0:
			self.lines = []
		else:
			self.lines = [LineEntry(i, line[1][0], line[0], line[1][1]) for i, line in enumerate(result)]

	def __len__(self):
		return len(self.lines)
	
	def __getitem__(self, key):
		return self.lines[key]

	# find string index using basic string compare and fallback to levenshtein distance
	def find(self, key, score_cutoff=0.5, strip=None, strip_flags = 0):
		# basic string compare
		for i, line in enumerate(self.lines):
			index = line.txt.find(key)
			if index != -1:
				return i
		
		# compare using levenshtein distance
		for i, line in enumerate(self.lines):
			str = line.txt
			if strip is not None:
				str = re.sub(strip, '', str, flags=strip_flags)

			r = ratio(str, key, score_cutoff=score_cutoff)
			if r >= score_cutoff:
				return i
			
		return -1

	# take lines while matching predicate, optionally remove from source
	def takewhile(self, predicate, remove=True):
		lines = []
		for i in self.lines:
			if predicate(i):
				lines.append(i)
			else:
				break

		if remove:
			self.lines = self.lines[len(lines):]

		return LineCollection(lines)

	# take lines between given cy offsets
	def takebetween(self, y_min, y_max, remove=True):
		lines = []
		for i in self.lines:
			if y_min <= i.cy and y_max >= i.cy:
				lines.append(i)

		if remove:
			for i in lines:
				self.lines.remove(i)

		return LineCollection(lines)

	# split line collection at offset (returning that entry, collection of preceding lines, collection of trailing lines)
	def splitat(self, index):
		if index < 0 or index > len(self.lines) - 1:
			return LineEntry.EMPTY(), LineCollection([]), LineCollection([])
		
		return self.lines[index], LineCollection(self.lines[0:index]), LineCollection(self.lines[index+1:])

# used in debug mode to navigate between images in a directory
class SimpleDirectoryNavigator():
	def __init__(self, source_path):
		self.dir_path = os.path.dirname(source_path)
		self.current_image_filename = os.path.basename(source_path)

	def getImagePath(self, offsetFromCurrent = 0):
		for root, dirs, files in os.walk(self.dir_path):
			imgs = list(filter(lambda x : x.endswith('.png') or x.endswith('.jpg'), files))
			for i, file in enumerate(imgs):
				if file == self.current_image_filename:
					index = i + offsetFromCurrent
					if index >= 0 and index < len(imgs):
						self.current_image_filename = imgs[index]
						return os.path.join(self.dir_path, self.current_image_filename), \
							index - 1 >= 0 and index - 1 < len(imgs), \
							index + 1 >= 0 and index + 1 < len(imgs)
					else:
						return None, \
							index - 1 >= 0 and index - 1 < len(imgs), \
							index + 1 >= 0 and index + 1 < len(imgs)
			break

class D4ItemTooltipOCR():
	def __init__(self):
		self.img_tmpl_affix = cv2.imread('templates/affix.png')
		self.img_tmpl_reroll = cv2.imread('templates/enchanted_rerolled.png')
		self.img_tmpl_aspect = cv2.imread('templates/inprint_aspect.png')
		self.img_tmpl_wstat = cv2.imread('templates/weapon_stat.png')
		self.img_tmpl_socket = cv2.imread('templates/socket.png')
		self.img_tmpl_socket_mask = cv2.imread('templates/socket_mask.png')
		
		self.templates = {
				'affix': self.img_tmpl_affix, 
				'reroll': self.img_tmpl_reroll, 
				'aspect': self.img_tmpl_aspect, 
				'wstat': self.img_tmpl_wstat,
				'socket': [self.img_tmpl_socket, self.img_tmpl_socket_mask]
				}

	def processImage(self, source_path, find_tooltip=True, debug=False):
		if debug == True:
			print(f'[Source image: \'{source_path}\']')
		
		input_image = cv2.imread(source_path)

		# find the item tooltip
		found_tooltip = False
		if find_tooltip == True:
			# preprocess input image
			height, width = input_image.shape[:2]
			hsv = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV) 
			tmp = cv2.inRange(hsv, np.array([69, 45, 47]), np.array([85, 106, 73]))
			
			kernel = np.ones((8,8), np.uint8)
			tmp = cv2.dilate(tmp, kernel, iterations = 5)

			# find contour most likely to be the tooltip
			tooltipcontours_image = input_image.copy()
			contours, hierarchy = cv2.findContours(tmp, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
			for cnt in contours:
				x, y, w, h = cv2.boundingRect(cnt)
				if cv2.contourArea(cnt) > 10000 and h > w and w < width * 0.3 and w > width * 0.15:
					expand = 15
					tooltip = input_image[y - expand:y + h + expand, x - expand:x + w + expand]
					found_tooltip = True

					cv2.rectangle(tooltipcontours_image, (x - expand, y - expand), (x + w + expand, y + h + expand), (0, 255, 0), 3)
				else:
					cv2.rectangle(tooltipcontours_image, (x, y), (x + w, y + h), (255, 0, 0), 3)
			
			tooltipcontours_image = cv2.resize(tooltipcontours_image, (0,0), fx = 0.5, fy = 0.5)
		else:
			tooltip = input_image

		if find_tooltip == True and found_tooltip != True:
			print("ERROR: Failed to find tooltip...")
			return None
		else:
			# use paddle ocr with custom d4 tooltip trained recognition model
			ocr = PaddleOCR(use_angle_cls=False, lang='en', show_log=False, det_db_unclip_ratio=2.0, rec_model_dir='paddleocr-models/en_PP-OCRv3_rec-d4_tooltip')
			result = ocr.ocr(tooltip, cls=False)

			if debug == True:
				# print ocr results
				print('OCR results:')
				for idx in range(len(result)):
					res = result[idx]
					for line in res:
						print(line)
				print('') # newline

			result = result[0]
			boxes = [line[0] for line in result]
			txts = [line[1][0] for line in result]
			scores = [line[1][1] for line in result]
			ocr_img = draw_ocr(tooltip, boxes, txts, scores, font_path='C:/Windows/fonts/Arial.ttf')

			# use scale invariant template matching to find all symbols denoting different types of tooltip lines
			data, templmatch_image = scaleInvariantMultiTemplateMatch(tooltip, self.templates, show_log=debug)
			
			if debug == True:
				cv2.imshow('ocr_img', ocr_img)
				if find_tooltip == True:
					cv2.imshow('tooltip_countours_img', tooltipcontours_image)
				cv2.imshow('template_match_img', templmatch_image)

			# =====================================================
			# build item tooltip data object for json serialization
			# =====================================================
			lc = LineCollection(result)

			# find item power and split on it (item power always exists in the tooltip)
			txt_item_power, before, after = lc.splitat(lc.find('Item Power', strip='^([\s\d\+]+)'))
			item_power, item_power_upgraded = txt_item_power.re_get('(?P<ip>\d+)(?:\+(?P<ipu>\d+))?', 'ip', 'ipu', flags=re.I)

			# spit on requires level because nothing below it is of interest
			txt_requires_level, after, tmp1 = after.splitat(after.find('Requires Level', strip='([\s\d]+)$'))

			# remove anything after last data entry (socket, aspect or affix)
			# strips flavor text from our collection
			if (len(data) >= 1):
				key, pt1, pt2, max_val = data[-1]
				after = after.takewhile(lambda x : x.tl[1] <= pt2[1] or x.tl[0] >= pt2[0] - int((pt2[0] - pt1[0]) / 2), remove=False)

			# item name is always in uppercase
			item_name = before.takewhile(lambda x : x.txt.upper() == x.txt)
			# item type follows item name and is the last line before item power
			item_type = before #before[len(item_name):]

			# upgrades follows item power (if it exists)
			txt_upgrades, *_ = lc.splitat(lc.find('Upgrades:', strip='([\s\d/]+)$'))
			item_upgrades_current, item_upgrades_max = txt_upgrades.re_get('(?P<upc>\d+)/(?P<upm>\d+)', 'upc', 'upm')

			item = {'affixes': [], 'stats': [], 'sockets': [] }
			item['name'] = lines_join(' ', [l.txt.strip() for l in item_name])
			item['type'] = lines_join(' ', [l.txt.strip() for l in item_type])
			item['item_power'] = item_power
			item['item_power_upgraded'] = item_power_upgraded
			item['item_upgrades_current'] = item_upgrades_current
			item['item_upgrades_max'] = item_upgrades_max

			if debug == True:
				print('Line cy offsets:')
				for a in after.lines:
					print(f'[cy: {a.cy}]: {a.txt}')
				print('') # newline

			# fetch lines belonging to any data entries using relative positions
			for i, d in enumerate(data):
				key_1, pt1_1, pt2_1, max_val_1 = d

				# anything above the first data entry is going to be weapon or armor stats
				stats = after.takebetween(0, pt1_1[1])
				for s in stats.lines:
					item['stats'].append(s.txt)

				# use the position of the next data entry to find the lines belonging to the current data entry
				if len(data) > i + 1:
					key_2, pt1_2, pt2_2, max_val_2 = data[i + 1]
				else:
					key_2, pt1_2, pt2_2, max_val_2 = None, [0, 9999], [0, 9999], 0.0

				dline = after.takebetween(pt1_1[1], pt2_2[1] - (pt2_2[1] - pt1_2[1]))
				if key_1 == 'affix' or key_1 == 'reroll':
					item['affixes'].append(lines_join(' ', [l.txt.strip() for l in dline or after]))
				elif key_1 == 'wstat':
					item['stats'].append(lines_join(' ', [l.txt.strip() for l in dline or after]))
				elif key_1 == 'aspect':
					item['aspect'] = lines_join(' ', [l.txt.strip() for l in dline or after])
				elif key_1 == 'socket':
					item['sockets'].append(lines_join(' ', [l.txt.strip() for l in dline or after]))
			
			# serialize to json
			jsonstr = json.dumps(item, sort_keys=True, indent=3)
			return jsonstr

# use substitution to fix common errors in recognized text
def lines_join(separator, iterable):
	result = separator.join(iterable)

	result = re.sub('(?<=[a-z])(?:\s+-\s+|\s+-|-\s+)(?=[a-z])', '-', result, flags=re.I) # [a -b]: missing space
	result = re.sub('^0\+', '+', result) # 0+: symbol recognized as 0
	result = re.sub('%([a-z])', r'% \1', result, flags=re.I) # %a: missing space
	result = re.sub(r'\b([A-Z])(([a-z]+)([A-Z]+)([a-z]*))\b', lambda m: m.group(1) + m.group(2).lower(), result) # PulveriZe: in word case mismatch
	result = re.sub('\]([A-Z])', r'] \1', result) # ]A: missing space
	result = re.sub('\[([\d\.]+)\s+([\d\.]+)\]%', r'[\1 - \2]%', result) # 15.0 -20.0: missing space

	return result

# scale invariant multi template matching using image pyramid
def scaleInvariantMultiTemplateMatch(img_source, img_tmpl_dict, matchTemplateThreshold=0.95, show_log=False):
	source = cv2.cvtColor(img_source, cv2.COLOR_BGR2GRAY)
	
	# pre-process templates
	tmpls = []
	tmpls_mask = []
	for k, tmpl in img_tmpl_dict.items():
		if isinstance(tmpl, list):
			tmpls.append(cv2.cvtColor(tmpl[0], cv2.COLOR_BGR2GRAY))
			tmpls_mask.append(cv2.cvtColor(tmpl[1], cv2.COLOR_BGR2GRAY))
		else:
			tmpls.append(cv2.cvtColor(tmpl, cv2.COLOR_BGR2GRAY))
			tmpls_mask.append(None)

	# per-downsampled source results
	results = []
	best_index = -1

	# template matching using 20 downsampled sources in scale range (0.2 - 1.0)
	for i, scale in enumerate(np.linspace(0.2, 1.0, 20)[::-1]):
		resized = cv2.resize(source, (0, 0), fx = scale, fy = scale)
		ratio = source.shape[1] / float(resized.shape[1])

		# per-template results
		t_results = {}
		max_value_high = 0

		# template match each input template
		for j, (k, img_tmpl) in enumerate(img_tmpl_dict.items()):
			tmpl = tmpls[j]
			tmpl_mask = tmpls_mask[j]

			# skip to next iteration if downsampled source is smaller than current template
			if resized.shape[0] < tmpl.shape[0] or resized.shape[1] < tmpl.shape[1]:
				t_results[k] = (None, None, 0.0, (0, 0), ratio)
				continue

			# match template and threshold with custom threshold
			result = cv2.matchTemplate(resized, tmpl, cv2.TM_CCORR_NORMED, mask=tmpl_mask)
			T, threshed = cv2.threshold(result, matchTemplateThreshold, 1., cv2.THRESH_TOZERO)

			# find best match score for current downsampled source in effort
			# to figure out which scale give the best overall results
			min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(threshed)
		
			t_results[k] = (result, threshed, max_val, max_loc)
			if max_val > max_value_high:
				max_value_high = max_val

		results.append((t_results, max_value_high, ratio))

		if best_index == -1:
			best_index = 0
		elif max_value_high > results[best_index][1]:
			best_index = i

	# best downsampled scale
	t_results, max_value_high, ratio = results[best_index]
	if show_log == True:
		print('Template matching:')
		print(f'best_index: {best_index}, r: {ratio}', end='\n\n')

	dest = img_source.copy()
	data = []

	# find best template match locations and draw them on a copy of source
	if show_log == True:
		print('Found templates:')

	for i, (k, img_tmpl) in enumerate(img_tmpl_dict.items()):
		h, w = tmpls[i].shape
		result, threshed, max_val, max_loc = t_results[k]
		
		while max_val > 0.9:
			pt1 = (int(round(max_loc[0] * ratio, 0)), int(round(max_loc[1] * ratio, 0)))
			pt2 = (int(round((max_loc[0] + w + 1) * ratio, 0)), int(round((max_loc[1] + h + 1) * ratio, 0)))

			if show_log == True:
				print(f'key: {k}, region: {pt1} => {pt2}, max_val: {max_val}')
			
			data.append((k, pt1, pt2, max_val))
			cv2.rectangle(dest, pt1, pt2, (0,255,0), 2)

			threshed[max_loc[1] - h // 2 : max_loc[1] + h // 2 + 1, max_loc[0] - w // 2 : max_loc[0] + w // 2 + 1] = 0   
			min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(threshed)

	if show_log == True and len(data) == 0:
		print('[None]', end='\n\n')
	elif show_log == True:
		print('') # newline

	# sort locations on y-offset in ascending order
	data.sort(key=lambda x : x[1][1])

	return data, dest

# create a tkinter window for viewing item tooltip data and
# navigating between images in the current directory
def showItemDataFrame(source_path, find_tooltip=True):
	dnav = SimpleDirectoryNavigator(source_path)
	it_ocr = D4ItemTooltipOCR()

	def toggle_button(btn, state):
		if state == True:
			btn["state"] = tk.NORMAL
		else:
			btn["state"] = tk.DISABLED

	def showImage(offsetFromCurrent = 0):
		image_path, has_prev, has_next = dnav.getImagePath(offsetFromCurrent)

		toggle_button(prev_button, has_prev)
		toggle_button(next_button, has_next)

		if image_path:
			jsonstr = it_ocr.processImage(image_path, find_tooltip, debug=True)

			win.title(f'D4 Item Data: {dnav.current_image_filename}')
			T.delete(1.0, tk.END)

			if jsonstr:
				T.insert(tk.END, jsonstr)
		else:
			win.title(f'D4 Item Data')
			T.delete(1.0, tk.END)
	
	win = tk.Tk()
	win.geometry("800x600")
	win.title("D4 Item Data")

	button_frame = tk.Frame(win)
	button_frame.pack(fill=tk.X, side=tk.BOTTOM)

	scroll_v = tk.Scrollbar(win)
	scroll_v.pack(side=tk.RIGHT,fill="y")
	scroll_h = tk.Scrollbar(win, orient=tk.HORIZONTAL)
	scroll_h.pack(side=tk.BOTTOM, fill= "x")

	T = tk.Text(win, height = 50, width = 500, yscrollcommand= scroll_v.set,xscrollcommand = scroll_h.set, wrap=tk.NONE,)
	T.config(font=("Courier New", 11))
	T.pack(fill=tk.BOTH, expand=0)

	scroll_h.config(command = T.xview)
	scroll_v.config(command = T.yview)
	
	prev_button = tk.Button(button_frame, text="Prev", command = lambda:showImage(-1))
	next_button = tk.Button(button_frame, text="Next", command = lambda:showImage(1))

	button_frame.columnconfigure(0, weight=1)
	button_frame.columnconfigure(1, weight=1)

	prev_button.grid(row=0, column=0, sticky=tk.W+tk.E)
	next_button.grid(row=0, column=1, sticky=tk.W+tk.E)

	win.after(0, showImage)
	tk.mainloop()

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--source-img', type=str, default='examples/screenshot_001.png', help='path to the source image')
	parser.add_argument('--json-output', type=str, default=None, help='output path for item tooltip json data')
	parser.add_argument('--find-tooltip', default=True, type=lambda x: x.lower() not in ['false', 'no', '0', 'None'], help='toggle find tooltip in source [true/false]')
	parser.add_argument('--debug', default=False, type=lambda x: x.lower() not in ['false', 'no', '0', 'None'], help='toggle debug mode [true/false]')
	opt = parser.parse_args()

	print("")
	print("======================================")
	print("Diablo IV: Item Tooltip OCR")
	print("======================================")
	print("")

	# check that source image exists
	if os.path.isfile(opt.source_img) == False:
		print('ERROR: Source image not found...', end='\n\n')
		parser.print_help()
		exit(1)

	# use debug mode
	if opt.debug == True:
		showItemDataFrame(opt.source_img, opt.find_tooltip)
		cv2.destroyAllWindows()

	# output json data from source
	else:
		it_ocr = D4ItemTooltipOCR()
		jsonstr = it_ocr.processImage(opt.source_img, opt.find_tooltip, debug=False)

		if (opt.json_output):
			with open(opt.json_output, 'w') as outfile:
				outfile.write(jsonstr)
			print(f'Item tooltip json written to \'{opt.json_output}\'.', end='\n\n')
		else:
			print(jsonstr)