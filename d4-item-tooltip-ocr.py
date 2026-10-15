import cv2
import numpy as np
import json
import re
import time
import os
import sys
import math
import argparse
from paddleocr import TextDetection

import paddle
from Levenshtein import ratio

class LineEntry():
	def __init__(self, index, txt, boxes, score, char_xs=None):
		self.index = index
		self.txt = txt
		self.score = score
		self.char_xs = char_xs or [] # approximate x-center of each character in txt
		x = [b[0] for b in boxes]
		y = [b[1] for b in boxes]
		self.tl = (min(x), min(y)) #top-left
		self.br = (max(x), max(y)) #bottom-right
		self.dx = self.br[0] - self.tl[0]
		self.dy = self.br[1] - self.tl[1]
		self.cy = self.tl[1] + int(self.dy / 2.0)
		self.a = self.dx * self.dy

	@staticmethod
	def EMPTY():
		return LineEntry(-1, None, [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], 0.0)

	def re_get(self, pattern: str, *return_group_names, flags = 0):
		if self.index == -1: return (*[None for gn in return_group_names],)

		m = re.search(pattern, self.txt, flags=flags)
		if m is None:
			return (*[None for gn in return_group_names],)
		results = [m.group(gn) for gn in return_group_names]
		return (*results,)

class LineCollection():
	def __init__(self, result):
		if isinstance(result, list) and all(isinstance(x, LineEntry) for x in result):
			self.lines = result
		elif isinstance(result, list) and len(result) == 0:
			self.lines = []
		else:
			self.lines = list(filter(lambda x : x.a > 150, [LineEntry(i, line[1][0], line[0], line[1][1], line[2] if len(line) > 2 else None) for i, line in enumerate(result)]))

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
	def __init__(self, source_path, run_tests_mode=False):
		self.run_tests_mode = run_tests_mode

		if run_tests_mode == False:
			self.dir_path = os.path.dirname(source_path)
			self.current_image_filename = os.path.basename(source_path)
		else:
			self.dir_path = source_path
			self.current_image_filename = None

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
	
	def getImagePath_RunTestsMode(self, offsetFromCurrent = 0):
		def isTestFile(x):
			if not (x.endswith('.png') or x.endswith('.jpg')):
				return False
			
			filename_without_ext = os.path.splitext(x)[0]
			json_path = os.path.join(self.dir_path, filename_without_ext + '.json')

			return os.path.isfile(json_path)

		for root, dirs, files in os.walk(self.dir_path):
			imgs = list(filter(isTestFile, files))
			for i, file in enumerate(imgs):
				if self.current_image_filename == None:
					index = i
					self.current_image_filename = imgs[index]
					return os.path.join(self.dir_path, self.current_image_filename), \
						index - 1 >= 0 and index - 1 < len(imgs), \
						index + 1 >= 0 and index + 1 < len(imgs)

				if file == self.current_image_filename:
					index = i + offsetFromCurrent
					if index >= 0 and index < len(imgs):
						self.current_image_filename = imgs[index]
						return os.path.join(self.dir_path, self.current_image_filename), \
							index - 1 >= 0 and index - 1 < len(imgs), \
							index + 1 >= 0 and index + 1 < len(imgs)
					else:
						return None
			break

class LegacyD4Recognizer():
	"""Recognize text with the legacy D4 PP-OCRv3 static inference model."""

	# ppocr/utils/en_dict.txt order: ASCII 0x30..0x7e, then 0x21..0x2f, then a
	# space. use_space_char=True appends a second space (the one the model was
	# trained to emit, as the label encoder maps ' ' to its last index).
	# CTC blank is index 0, giving 97 output classes.
	CHARACTER = ''.join(chr(c) for c in range(0x30, 0x7f)) + ''.join(chr(c) for c in range(0x21, 0x30)) + '  '

	def __init__(self, model_dir, batch_size=10, enable_mkldnn=True):
		self.model_dir = os.path.abspath(model_dir)
		self.batch_size = max(1, int(batch_size))
		self.image_shape = (3, 48, 320)
		model_file = os.path.join(self.model_dir, 'inference.pdmodel')
		params_file = os.path.join(self.model_dir, 'inference.pdiparams')
		if not os.path.isfile(model_file):
			raise FileNotFoundError(f'Legacy D4 recognition model not found: {model_file}')
		if not os.path.isfile(params_file):
			raise FileNotFoundError(f'Legacy D4 recognition parameters not found: {params_file}')

		config = paddle.inference.Config(model_file, params_file)
		config.disable_gpu()
		if enable_mkldnn:
			try:
				config.set_mkldnn_cache_capacity(10)
				config.enable_mkldnn()
			except Exception as exc:
				print(f'WARNING: Could not enable oneDNN for legacy recognizer: {exc}', file=sys.stderr)
		if hasattr(config, 'enable_new_ir'):
			try:
				config.enable_new_ir()
				config.enable_new_executor()
			except Exception:
				pass
		config.disable_glog_info()
		self.predictor = paddle.inference.create_predictor(config)
		self.input_names = list(self.predictor.get_input_names())
		self.output_names = list(self.predictor.get_output_names())
		if not self.input_names or not self.output_names:
			raise RuntimeError(f'Legacy D4 model tensors unavailable (inputs={self.input_names}, outputs={self.output_names})')
		print(f'Legacy D4 recognizer: input={self.input_names}, outputs={self.output_names}', file=sys.stderr)

	def preprocess(self, img, max_wh_ratio):
		# Same as PaddleOCR 2.x TextRecognizer.resize_norm_img, which the model
		# was trained with: the input width grows with the widest crop of the
		# batch, so long lines are not squeezed into 320px.
		img_c, img_h, img_w = self.image_shape
		if img is None or img.size == 0:
			raise ValueError('Cannot recognize an empty text crop.')
		img_w = int(img_h * max_wh_ratio)
		h, w = img.shape[:2]
		ratio = w / float(max(h, 1))
		resized_w = min(img_w, max(1, int(math.ceil(img_h * ratio))))
		resized = cv2.resize(img, (resized_w, img_h))
		resized = resized.astype('float32').transpose((2, 0, 1)) / 255.0
		resized = (resized - 0.5) / 0.5
		padding = np.zeros((img_c, img_h, img_w), dtype=np.float32)
		padding[:, :, :resized_w] = resized
		return padding

	@staticmethod
	def crop_from_quad(image, points):
		points = np.asarray(points, dtype=np.float32).reshape(4, 2)
		w = max(1, int(max(np.linalg.norm(points[0] - points[1]), np.linalg.norm(points[2] - points[3]))))
		h = max(1, int(max(np.linalg.norm(points[0] - points[3]), np.linalg.norm(points[1] - points[2]))))
		dst = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
		m = cv2.getPerspectiveTransform(points, dst)
		crop = cv2.warpPerspective(image, m, (w, h), borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC)
		if crop.shape[0] / float(max(crop.shape[1], 1)) >= 1.5:
			crop = np.rot90(crop)
		return crop

	@classmethod
	def decode_ctc(cls, output):
		arr = np.asarray(output)
		if arr.ndim != 3:
			raise RuntimeError(f'Expected rank-3 CTC output, got shape {arr.shape}.')
		idx = arr.argmax(axis=2)
		prob = arr.max(axis=2)
		results = []
		for b in range(arr.shape[0]):
			chars = []
			conf = []
			steps = []
			last = -1
			for t, token in enumerate(idx[b]):
				token = int(token)
				if token == 0:
					last = token
					continue
				if token == last:
					continue
				last = token
				ci = token - 1
				if 0 <= ci < len(cls.CHARACTER):
					chars.append(cls.CHARACTER[ci])
					conf.append(float(prob[b, t]))
					steps.append(t)
			results.append((''.join(chars), float(np.mean(conf)) if conf else 1.0, steps))
		return results

	def recognize(self, crops):
		# Returns (text, score, char_xs) per crop, where char_xs holds the
		# approximate horizontal center of each character in crop pixels.
		# Batch crops of similar aspect ratio together (as PaddleOCR 2.x does) so
		# short lines are not padded out to the width of the longest line.
		img_c, img_h, img_w = self.image_shape
		wh_ratios = [c.shape[1] / float(max(c.shape[0], 1)) for c in crops]
		order = np.argsort(wh_ratios)
		results = [None] * len(crops)
		for offset in range(0, len(crops), self.batch_size):
			chunk = order[offset:offset + self.batch_size]
			max_wh_ratio = max([img_w / float(img_h)] + [wh_ratios[i] for i in chunk])
			batch = np.stack([self.preprocess(crops[i], max_wh_ratio) for i in chunk], axis=0).astype(np.float32)
			in_handle = self.predictor.get_input_handle(self.input_names[0])
			in_handle.reshape(batch.shape)
			in_handle.copy_from_cpu(batch)
			self.predictor.run()
			outputs = [(name, self.predictor.get_output_handle(name).copy_to_cpu()) for name in self.output_names]
			ctc_candidates = [(name, value) for name, value in outputs if np.asarray(value).ndim == 3]
			if not ctc_candidates:
				raise RuntimeError(f'No rank-3 CTC output found. Outputs: {[(n, np.asarray(v).shape) for n, v in outputs]}')
			ctc = ctc_candidates[0][1]
			# each CTC timestep covers a fixed slice of the (resized) input width
			step_w = batch.shape[3] / float(ctc.shape[1])
			for i, (text, score, steps) in zip(chunk, self.decode_ctc(ctc)):
				crop_h, crop_w = crops[i].shape[:2]
				resized_w = min(batch.shape[3], max(1, int(math.ceil(img_h * wh_ratios[i]))))
				scale = crop_w / float(resized_w)
				results[i] = (text, score, [(t + 0.5) * step_w * scale for t in steps])
		return results


class D4ItemTooltipOCR():
	def __init__(self, rec_model_dir='paddleocr-models/en_PP-OCRv3_rec-d4_tooltip', det_model_name='PP-OCRv4_mobile_det'):
		# PaddleOCR 3.x provides text detection only. The custom D4 recognition
		# network is a legacy static export, so load it directly via Paddle Inference.
		# PaddlePaddle 3.3.0/3.3.1 has a CPU oneDNN/PIR bug for some OCR
		# detection models (ArrayAttribute<DoubleAttribute>). Disable oneDNN for
		# the PaddleOCR 3.x detector; use the standard CPU inference path instead.
		self.detector = TextDetection(
			model_name=det_model_name,
			device='cpu',
			enable_mkldnn=False,
			# box expansion of PaddleOCR 2.x; tighter boxes clip glyphs
			unclip_ratio=2.0)
		self.recognizer = LegacyD4Recognizer(rec_model_dir, batch_size=10, enable_mkldnn=True)
		self.img_tmpl_affix = cv2.imread('templates/affix.png')
		self.img_tmpl_reroll = cv2.imread('templates/enchanted_rerolled.png')
		self.img_tmpl_aspect = cv2.imread('templates/inprint_aspect.png')
		self.img_tmpl_wstat = cv2.imread('templates/weapon_stat.png')
		self.img_tmpl_socket = cv2.imread('templates/socket.png')
		self.img_tmpl_socket_mask = cv2.imread('templates/socket_mask_new.png')
		self.templates = {
			'affix': self.img_tmpl_affix,
			'reroll': self.img_tmpl_reroll,
			'aspect': self.img_tmpl_aspect,
			'wstat': self.img_tmpl_wstat,
			'socket': [self.img_tmpl_socket, self.img_tmpl_socket_mask]
			}

	@staticmethod
	def _result_field(result, name, default=None):
		if hasattr(result, name):
			return getattr(result, name)
		try:
			return result[name]
		except (KeyError, TypeError, IndexError):
			return default

	@classmethod
	def _extract_detection_boxes(cls, result):
		if result is None:
			return [], []
		res = result[0] if isinstance(result, (list, tuple)) and result else result
		boxes = cls._result_field(res, 'dt_polys', [])
		scores = cls._result_field(res, 'dt_scores', [])
		boxes = list(boxes) if boxes is not None else []
		scores = list(scores) if scores is not None else []
		return [np.asarray(b, dtype=np.float32) for b in boxes], [float(s) for s in scores]

	@staticmethod
	def _draw_ocr_debug(image, boxes, txts, scores):
		out = image.copy()
		for box, txt, score in zip(boxes, txts, scores):
			pts = np.asarray(box, dtype=np.int32).reshape(-1, 1, 2)
			if pts.shape[0] >= 2:
				cv2.polylines(out, [pts], True, (0, 255, 0), 1)
			anchor = tuple(pts[0, 0])
			cv2.putText(out, f'{txt} ({float(score):.2f})', (anchor[0], max(12, anchor[1] - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1, cv2.LINE_AA)
		return out

	def processImage(self, source_path, find_tooltip=True, debug=False):
		start_pi = time.time()
		jsonstr, ocr_deltatime = self.processImage_internal(source_path, find_tooltip, debug)
		end_pi = time.time()
		print(f'Total: {end_pi-start_pi:.2f}s, OCR: {ocr_deltatime:.2f}s', end='\n\n')

		return jsonstr
	
	def processImage_internal(self, source_path, find_tooltip=True, debug=False):
		if debug == True:
			print(f'[Source image: \'{source_path}\']')
		
		# Load the source image robustly.  cv2.imread() can return None even when
		# os.path.isfile() succeeds (for example, if the image cannot be decoded by
		# the installed OpenCV build).  Try the byte-oriented decoder as a fallback
		# and emit a useful diagnostic instead of crashing on .shape below.
		input_image = cv2.imread(source_path)
		if input_image is None:
			try:
				encoded = np.fromfile(source_path, dtype=np.uint8)
				if encoded.size > 0:
					input_image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
			except (OSError, ValueError) as exc:
				print(f"ERROR: Failed to read source image '{source_path}': {exc}", file=sys.stderr)

		if input_image is None:
			try:
				size = os.path.getsize(source_path)
			except OSError:
				size = 'unknown'
			print(
				f"ERROR: Failed to decode source image '{source_path}' (size: {size} bytes).",
				file=sys.stderr
			)
			print(
				"WARNING: No OCR or tooltip detection was attempted because the source image could not be decoded.",
				file=sys.stderr
			)
			return None, 0.0

		# find the item tooltip
		found_tooltip = False
		if find_tooltip == True:
			# Detect the tooltip frame from its blue/cyan border.  The character
			# panel can have another frame immediately above the tooltip (the
			# "EQUIPPED" header), so there may be several horizontal candidates.
			# We select the lowest strong horizontal frame in the upper part of
			# the screen; internal separators are much less saturated and do not
			# normally pass this mask.
			height, width = input_image.shape[:2]
			hsv = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)
			tmp = cv2.inRange(hsv, np.array([69, 45, 47]), np.array([85, 106, 73]))

			tooltipcontours_image = input_image.copy()

			# Find the two long vertical frame bands.
			column_counts = np.count_nonzero(tmp, axis=0)
			column_threshold = max(12, int(height * 0.08))
			vertical_pixels = np.where(column_counts >= column_threshold)[0]

			vertical_groups = []
			if len(vertical_pixels) > 0:
				start_x = previous = int(vertical_pixels[0])
				for x in vertical_pixels[1:]:
					x = int(x)
					if x <= previous + 1:
						previous = x
					else:
						vertical_groups.append((start_x, previous, int(column_counts[start_x:previous + 1].max())))
						start_x = previous = x
				vertical_groups.append((start_x, previous, int(column_counts[start_x:previous + 1].max())))

			# Evaluate pairs by frame evidence instead of their width as a fraction
			# of the full screenshot.  That ratio changes dramatically between a
			# cropped tooltip sample and a full gameplay screenshot.
			pair = None
			pair_top_y = None
			best_pair_score = float('-inf')
			for i in range(len(vertical_groups)):
				for j in range(i + 1, len(vertical_groups)):
					left = vertical_groups[i]
					right = vertical_groups[j]
					x_left = (left[0] + left[1]) // 2
					x_right = (right[0] + right[1]) // 2
					separation = x_right - x_left
					# Ignore adjacent marks; actual tooltip side rails are distinct
					# vertical structures. This is an absolute pixel guard, not a
					# screen-width percentage.
					if separation < 80:
						continue

					left_ys = np.where(np.any(tmp[:, left[0]:left[1] + 1] > 0, axis=1))[0]
					right_ys = np.where(np.any(tmp[:, right[0]:right[1] + 1] > 0, axis=1))[0]
					if len(left_ys) == 0 or len(right_ys) == 0:
						continue

					# The relevant top edge should occur around where both side rails
					# become active. In full-screen captures, the tooltip can be well
					# below 25% of the image height, so search relative to the rails,
					# not within a fixed screen-height band.
					side_start_y = max(int(left_ys[0]), int(right_ys[0]))
					row_left = max(0, x_left - 12)
					row_right = min(width, x_right + 13)
					row_counts = np.count_nonzero(tmp[:, row_left:row_right], axis=1)
					row_threshold = max(12, min(int(separation * 0.12), 100))
					horizontal_pixels = np.where(row_counts >= row_threshold)[0]
					horizontal_groups = []
					if len(horizontal_pixels) > 0:
						start_y = previous_y = int(horizontal_pixels[0])
						for y in horizontal_pixels[1:]:
							y = int(y)
							if y <= previous_y + 1:
								previous_y = y
							else:
								horizontal_groups.append((start_y, previous_y, int(row_counts[start_y:previous_y + 1].max())))
								start_y = previous_y = y
						horizontal_groups.append((start_y, previous_y, int(row_counts[start_y:previous_y + 1].max())))

					# The visible horizontal top border does not always begin at the
					# same y-coordinate as the blue side-rail mask. On dark/gold UI themes,
					# the side rails can have a few pixels of blue trim below the top edge.
					# Allow a modest lead-in before the side rails, scaled to the measured
					# rail separation (not the full screenshot dimensions).
					search_before = max(32, int(separation * 0.08))
					search_ahead = max(70, int(separation * 0.20))
					candidates = [g for g in horizontal_groups
						if side_start_y - search_before <= g[0] <= side_start_y + search_ahead]
					if not candidates:
						continue
					# Prefer an edge at/just after the side rails begin. When the same
					# rails continue through the EQUIPPED header, its upper border can also
					# fall in the expanded search window; choosing that earlier edge would
					# pull the header into the item crop. Only use a preceding edge when
					# there is no edge aligned with or below the rail start (as on the newer
					# gold-framed tooltip where the top border is slightly above blue trim).
					aligned_or_after = [g for g in candidates if g[0] >= side_start_y - 2]
					if aligned_or_after:
						top_edge = min(aligned_or_after, key=lambda g: (g[0], -g[2]))
					else:
						top_edge = min(candidates, key=lambda g: (abs(g[0] - side_start_y), -g[2]))

					left_max = int(left[2])
					right_max = int(right[2])
					vertical_overlap = max(
						0,
						min(int(left_ys[-1]), int(right_ys[-1])) - max(int(left_ys[0]), int(right_ys[0]))
					)
					distance_from_rails = top_edge[0] - side_start_y
					pair_score = (
						min(left_max, right_max)
						+ 0.25 * max(left_max, right_max)
						+ 0.75 * top_edge[2]
						+ 0.10 * min(vertical_overlap, 500)
						- 0.25 * distance_from_rails
					)
					if pair_score > best_pair_score:
						best_pair_score = pair_score
						pair = (x_left, x_right)
						pair_top_y = top_edge[0]

			if pair is None:
				if len(vertical_groups) >= 2:
					print(
						"WARNING: Tooltip side rails were found, but no top-edge candidate fell within the adaptive search window. Check the screenshot theme/scale.",
						file=sys.stderr
					)
				else:
					print("WARNING: No strong tooltip frame side bands were detected.", file=sys.stderr)

			if pair is not None and pair_top_y is not None:
				x_left, x_right = pair
				inner_margin = max(5, int(round((x_right - x_left) * 0.015)))
				x1 = max(0, x_left + inner_margin)
				x2 = min(width, x_right - inner_margin)
				y1 = max(0, pair_top_y + inner_margin + 2)
				# Do not assume a fixed tooltip height: effects and wrapped affixes
				# vary in length. The OCR parser stops at the tooltip's level line.
				y2 = height
				if x2 > x1 and y2 > y1:
					tooltip = input_image[y1:y2, x1:x2]
					found_tooltip = True
					if debug == True:
						print(f'Tooltip frame: x={x_left}:{x_right}, top={pair_top_y}; crop x={x1}:{x2}, y={y1}:{y2}')
					cv2.rectangle(tooltipcontours_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
			# Debug image: show the selected side rails used for detection.
			if pair is not None:
				cv2.line(tooltipcontours_image, (pair[0], 0), (pair[0], height - 1), (255, 0, 0), 2)
				cv2.line(tooltipcontours_image, (pair[1], 0), (pair[1], height - 1), (255, 0, 0), 2)

			tooltipcontours_image = cv2.resize(tooltipcontours_image, (0, 0), fx=0.5, fy=0.5)
		else:
			tooltip = input_image

		if find_tooltip == True and found_tooltip != True:
			print("ERROR: Failed to find tooltip...", file=sys.stderr)
			return None, 0.0
		else:
			start_ocr = time.time()
			try:
				det_result = self.detector.predict(tooltip, batch_size=1)
				det_boxes, det_scores = self._extract_detection_boxes(det_result)
			except Exception as exc:
				print(f"ERROR: PaddleOCR 3.x text detection failed: {exc}", file=sys.stderr)
				return None, (time.time() - start_ocr)

			filtered = [(box, score) for box, score in zip(det_boxes, det_scores) if score >= 0.30 and len(box) == 4]
			filtered = merge_row_fragments(filtered)
			filtered.sort(key=lambda item: (float(np.min(item[0][:, 1])), float(np.min(item[0][:, 0]))))
			crops = [self.recognizer.crop_from_quad(tooltip, box) for box, _ in filtered]
			try:
				rec_results = self.recognizer.recognize(crops)
			except Exception as exc:
				print(f"ERROR: Legacy D4 text recognition failed: {exc}", file=sys.stderr)
				return None, (time.time() - start_ocr)
			end_ocr = time.time()

			if len(rec_results) != len(filtered):
				print(f"ERROR: Recognition returned {len(rec_results)} results for {len(filtered)} detected text boxes.", file=sys.stderr)
				return None, (end_ocr - start_ocr)

			# A full gameplay screenshot includes the tooltip action footer and the
			# inventory below it. Those regions are outside the item text panel and
			# should not be included in the OCR crop. Use the recognized Durability
			# line as a content-based bottom boundary instead of guessing from the
			# screenshot height. This works for both full-screen and pre-cropped input.
			durability_hits = []
			for idx, ((box, _det_score), (text, _rec_score, _char_xs)) in enumerate(zip(filtered, rec_results)):
				if re.search(r'durab', text, flags=re.I):
					box_arr = np.asarray(box, dtype=np.float32)
					if box_arr.ndim == 2 and box_arr.shape[1] >= 2 and box_arr.shape[0] >= 4:
						line_top = int(np.floor(np.min(box_arr[:, 1])))
						line_bottom = int(np.ceil(np.max(box_arr[:, 1])))
						# Ignore spurious matches near the top (e.g. a malformed label).
						if line_bottom > int(tooltip.shape[0] * 0.20):
							durability_hits.append((line_bottom, line_top, idx, text))

			if durability_hits:
				# The bottom-most match is the most likely actual durability row.
				line_bottom, line_top, _durability_idx, durability_text = max(durability_hits, key=lambda hit: hit[0])
				line_height = max(1, line_bottom - line_top + 1)
				bottom_padding = max(8, int(round(line_height * 0.45)))
				crop_bottom = min(tooltip.shape[0], line_bottom + bottom_padding)
				if crop_bottom < tooltip.shape[0]:
					old_height = tooltip.shape[0]
					tooltip = tooltip[:crop_bottom, :].copy()
					kept_indices = [
						i for i, ((box, _det_score), _rec) in enumerate(zip(filtered, rec_results))
						if int(np.min(np.asarray(box)[:, 1])) < crop_bottom
					]
					filtered = [filtered[i] for i in kept_indices]
					rec_results = [rec_results[i] for i in kept_indices]
					if debug:
						print(f'Tooltip crop trimmed after {durability_text!r}: height {old_height} -> {crop_bottom}px')
			elif debug:
				# jewelry has no durability; parsing stops at Requires Level anyway
				print(
					"WARNING: Could not recognize the Durability line; keeping the detected crop height. "
					"The OCR image may include the tooltip action footer. Check --debug output if parsing is poor.",
					file=sys.stderr
				)

			# character positions are relative to the crop, which starts at the box's left edge
			ocr_lines = [[box.tolist(), [text, rec_score], [float(np.min(box[:, 0])) + x for x in char_xs]]
				for (box, _), (text, rec_score, char_xs) in zip(filtered, rec_results)]
			# drop empty and unreliable results (e.g. an icon detected as a separate text box)
			ocr_lines = [line for line in ocr_lines if line[1][0] and line[1][0].strip() and line[1][1] >= 0.5]
			if not ocr_lines:
				print("ERROR: PaddleOCR detected no recognizable text lines.", file=sys.stderr)
				return None, (end_ocr - start_ocr)

			if debug:
				boxes = [line[0] for line in ocr_lines]
				txts = [line[1][0] for line in ocr_lines]
				scores = [line[1][1] for line in ocr_lines]
				print('OCR results:')
				for line in ocr_lines:
					print(line[:2])
				print('')

			data, templmatch_image = scaleInvariantMultiTemplateMatch(tooltip, self.templates, show_log=debug)
			if debug:
				boxes = [line[0] for line in ocr_lines]
				txts = [line[1][0] for line in ocr_lines]
				scores = [line[1][1] for line in ocr_lines]
				ocr_img = self._draw_ocr_debug(tooltip, boxes, txts, scores)
				cv2.imshow('ocr_img', ocr_img)
				if find_tooltip == True:
					cv2.imshow('tooltip_countours_img', tooltipcontours_image)
				cv2.imshow('template_match_img', templmatch_image)

			lc = LineCollection(ocr_lines)
			for line in lc.lines:
				fix_letter_case(line, tooltip)

			# find item power and split on it (item power always exists in the tooltip)
			txt_item_power, before, after = lc.splitat(lc.find('Item Power', strip=r'^([\s\d\+]+)'))
			item_power, item_power_upgraded = txt_item_power.re_get(r'(?P<ip>\d+)(?:\+(?P<ipu>\d+))?', 'ip', 'ipu', flags=re.I)

			# spit on requires level because nothing below it is of interest
			txt_requires_level, after, tmp1 = after.splitat(after.find('Requires Level', strip=r'([\s\d]+)$'))

			# item name is always in uppercase
			# drop the EQUIPPED banner and small boxes from frame ornaments above the name
			header_h = np.median([l.dy for l in before.lines]) if before.lines else 0
			before.lines = [l for l in before.lines if l.txt.strip().lower() not in ('equipped', 'armory loadout') and l.dy >= 0.7 * header_h]
			item_name = before.takewhile(lambda x : x.txt.upper() == x.txt)
			# item type follows item name and is the last line before item power
			item_type = before #before[len(item_name):]

			# upgrades follows item power (if it exists)
			txt_upgrades, *_ = lc.splitat(lc.find('Upgrades:', strip=r'([\s\d/]+)$'))
			item_upgrades_current, item_upgrades_max = txt_upgrades.re_get(r'(?P<upc>\d+)/(?P<upm>\d+)', 'upc', 'upm')
			if txt_upgrades.index != -1:
				after.lines.remove(txt_upgrades)

			item = {'affixes': [], 'stats': [], 'sockets': [], 'aspect': None }
			item['name'] = lines_join(' ', [l.txt.strip() for l in item_name])
			# an icon next to the name is sometimes recognized as a trailing 'O' or '%'
			item['name'] = re.sub(r'(?:\s+\S|[^A-Za-z\s]+)$', '', item['name'])
			item['name'] = re.sub(r'\bOE\b', 'OF', item['name']) # F recognized as E in the name font
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
			# Group the remaining lines into tooltip entries. Every affix, aspect and
			# socket starts with an icon in a left gutter, and its text (including
			# wrapped lines) is indented past that gutter. Stats above the entries and
			# flavor text below them are flush with the left edge. Entry starts are
			# derived from this layout; template matches only decide the entry kind,
			# as the icons change too often between game versions to rely on them.
			ui_lines = ('armory loadout', 'equipped')
			lines = sorted([l for l in after.lines if l.txt.strip().lower() not in ui_lines], key=lambda l: l.cy)
			all_lines = [l for l in lc.lines if l.dy > 0]
			line_h = float(np.median([l.dy for l in all_lines])) if all_lines else 1.0
			left_x = float(np.percentile([l.tl[0] for l in all_lines], 10)) if all_lines else 0.0
			indent_x = left_x + 0.6 * line_h

			def text_x(line):
				# left edge of the first recognized character (the icon itself is
				# usually not recognized as text, even if it is inside the box)
				if line.char_xs:
					return max(line.tl[0], line.char_xs[0] - 0.25 * line_h)
				return line.tl[0]

			def icon_extent(line, tx):
				# Look for an icon left of the text: a blob of bright pixels that is
				# separated from the text by a gap. Thin vertical lines (weapon stat
				# tree connectors) and characters the recognizer missed (which touch
				# the text) do not count. Returns the icon's (x_start, x_end) or None.
				x1, x2 = int(max(0, left_x - 0.3 * line_h)), int(min(tooltip.shape[1], tx + 0.5 * line_h))
				y1 = int(max(0, line.tl[1] + 0.2 * line.dy))
				y2 = int(min(tooltip.shape[0], line.br[1] - 0.2 * line.dy))
				if x2 <= x1 or y2 <= y1:
					return None
				ink = np.count_nonzero(tooltip[y1:y2, x1:x2].max(axis=2) > 100, axis=0)
				runs = []
				for x in np.flatnonzero(ink):
					if runs and x - runs[-1][1] <= 0.15 * line_h:
						runs[-1][1] = x
					else:
						runs.append([x, x])
				text_start = tx + 0.1 * line_h - x1
				icon = None
				for start, end in runs:
					if end >= text_start:
						break
					if end - start + 1 >= 0.2 * line_h and ink[start:end + 1].sum() >= 0.03 * line_h * line_h:
						icon = (x1 + start, x1 + end)
				return icon

			def template_at(line, tx):
				for key, pt1, pt2, _max_val in data:
					cy = (pt1[1] + pt2[1]) / 2.0
					cx = (pt1[0] + pt2[0]) / 2.0
					if line.tl[1] - 0.3 * line_h <= cy <= line.br[1] + 0.3 * line_h and cx < tx:
						return key
				return None

			def glued_icon_token(line):
				# An icon is sometimes recognized as a single character glued to the
				# text, such as "gLucky Hit" (aspect star) or "-0.90 Attacks" (weapon
				# stat tree). The first two characters of a word are at most ~0.67 line
				# heights apart, while a glued icon is >= 0.75 line heights away.
				# Returns the index of the first real character if that is the case.
				if len(line.char_xs) > 1 and line.txt[1] != ' ' and line.char_xs[1] - line.char_xs[0] >= 0.72 * line_h:
					return 1
				return None

			def spaced_icon_token(line, next_line):
				# A short junk token separated by a space ("C) PocQue"), or symbols
				# without one ("@)PocQue"), in front of flush-left text. Returns the
				# index of the first real character.
				m = re.match(r'\s*(?:\S{1,2} +|[^\w\s]{1,2})(?=\w)', line.txt)
				if m is None or m.end() >= len(line.char_xs):
					return None
				k = m.end()
				if line.char_xs[k] - 0.25 * line_h < indent_x:
					return None
				if is_socket_text(line.txt[k:]):
					return k
				token = line.txt[:k].strip()
				if re.fullmatch(r'["\'(]?[A-Z]?[a-z]+[,.]?|[^\w]*\d[\d\W]*', token):
					return None
				# "He is ..." at the start of flavor text has the same shape as "C) PocQue",
				# so a token with letters or digits also needs an indented next line
				if not re.search(r'[A-Za-z0-9]', token) or (next_line is not None and text_x(next_line) >= indent_x):
					return k
				return None

			item_has_aspect = re.search(r'\b(legendary|unique|mythic)\b', item['type'], flags=re.I) is not None

			def is_socket_text(text):
				return re.search(r'\bempty\s+socket\b', text, flags=re.I) is not None

			entries = []
			current = None
			# drop icons detected as separate (small) text boxes and junk such as a lone "-"
			lines = [l for l in lines if re.search(r'[A-Za-z0-9]', l.txt) and l.dy >= 0.7 * line_h]
			for i, l in enumerate(lines):
				next_line = lines[i + 1] if i + 1 < len(lines) else None
				txt = l.txt.strip()
				tx = text_x(l)
				strip_at = glued_icon_token(l)
				if strip_at is None and tx < indent_x:
					strip_at = spaced_icon_token(l, next_line)
				if strip_at is not None:
					txt = l.txt[strip_at:].strip()
					tx = l.char_xs[strip_at] - 0.25 * line_h
				key = template_at(l, tx)
				if tx >= indent_x:
					# the hollow empty socket icon is too dark to be found as an icon
					icon = icon_extent(l, tx)
					is_start = key is not None or icon is not None or is_socket_text(txt) or strip_at is not None
				else:
					icon = None
					is_start = False

				if debug:
					icon_w = f'{(icon[1] - icon[0] + 1) / line_h:.2f}h' if icon else '-'
					pitch = f'{(l.char_xs[1] - l.char_xs[0]) / line_h:.2f}h' if len(l.char_xs) > 1 else '-'
					print(f'[x: {l.tl[0]:.0f}, text x: {tx:.0f}, indent x: {indent_x:.0f}, start: {is_start}, icon: {icon_w}, first pitch: {pitch}]: {l.txt}')

				if is_start:
					icon_w = (icon[1] - icon[0] + 1) / line_h if icon else 0.0
					current = {'key': key, 'lines': [l], 'texts': [txt], 'text_x': tx, 'icon_w': icon_w}
					entries.append(current)
				elif tx >= indent_x and current is not None and abs(tx - current['text_x']) < 0.6 * line_h:
					# wrapped line of the current entry
					current['lines'].append(l)
					current['texts'].append(txt)
				elif not entries:
					# anything before the first entry is a weapon or armor stat
					item['stats'].append(lines_join(' ', [txt]))
				else:
					# flush-left text after the entries is flavor text
					break

			def aspect_color_score(entry):
				# Legendary/Unique effect text is orange, Mythic Unique effect text is
				# purple. Score the OCR boxes themselves so dark background pixels do
				# not dominate the result (the Mythic tooltip background is a dark purple).
				colored_px = total_px = 0
				for line in entry['lines']:
					x1, y1 = int(max(0, line.tl[0])), int(max(0, line.tl[1]))
					x2, y2 = int(min(tooltip.shape[1], line.br[0] + 1)), int(min(tooltip.shape[0], line.br[1] + 1))
					if x2 <= x1 or y2 <= y1:
						continue
					h, sat, val = cv2.split(cv2.cvtColor(tooltip[y1:y2, x1:x2], cv2.COLOR_BGR2HSV))
					orange = (h >= 3) & (h <= 25) & (sat >= 110) & (val >= 90)
					purple = (h >= 135) & (h <= 160) & (sat >= 40) & (val >= 140)
					colored_px += np.count_nonzero(orange | purple)
					total_px += h.size
				return colored_px / float(total_px) if total_px else 0.0

			# classify entries: template matches first, then icon size, layout and
			# color. Icon widths relative to the line height: affix and aspect
			# markers are ~0.3-0.6, weapon stat tree connectors ~0.75-0.85 and
			# socketed gems/runes ~0.95-1.1. The aspect follows the affixes, and
			# sockets follow the aspect.
			has_aspect_template = any(e['key'] == 'aspect' for e in entries)
			seen_aspect = seen_affix = False
			for e in entries:
				text = lines_join(' ', e['texts'])
				if e['key'] == 'socket' or is_socket_text(text) or (e['key'] != 'wstat' and e['icon_w'] >= 0.9):
					kind = 'socket'
				elif e['key'] == 'wstat' or (not seen_affix and not seen_aspect and e['icon_w'] >= 0.65):
					kind = 'stat'
				elif seen_aspect:
					kind = 'socket'
				elif e['key'] == 'aspect' or (item_has_aspect and not has_aspect_template and aspect_color_score(e) >= 0.02):
					kind = 'aspect'
					seen_aspect = True
				else:
					kind = 'affix'
					seen_affix = True

				if debug:
					print(f'[{kind} / {e["key"]}, icon: {e["icon_w"]:.2f}h, aspect color: {aspect_color_score(e):.3f}]: {text}')

				if kind == 'aspect':
					item['aspect'] = text
				else:
					item[{'affix': 'affixes', 'stat': 'stats', 'socket': 'sockets'}[kind]].append(text)

			if debug:
				print('')

			# serialize to json
			jsonstr = json.dumps(item, sort_keys=True, indent=3)
			return jsonstr, (end_ocr - start_ocr)

# Some letters (C, O, S, V, W, X, Z) only differ in size from their lowercase
# version, and the recognizer often gets the case wrong ("40% Chance", "Curse
# On enemies"). Decide by comparing the top of the glyph in the image with the
# top of the letters of known case in the same line. V, W and X are left out,
# as their center column does not reach the top of the glyph.
X_HEIGHT_CHARS = set('acemnorsuvwxz')
CAP_HEIGHT_CHARS = set('ABDEFGHIJKLMNPQRTUYbdfhklt0123456789')
def fix_letter_case(line, image):
	if len(line.char_xs) != len(line.txt) or line.dy <= 0:
		return
	y1, y2 = int(max(0, line.tl[1])), int(min(image.shape[0], line.br[1]))
	if y2 - y1 < 4:
		return
	patch = image[y1:y2].max(axis=2).astype(np.float32)
	h, background = y2 - y1, float(np.median(patch[:, int(max(0, line.tl[0])):int(line.br[0])]))

	def glyph_top(cx):
		col = patch[:, int(max(0, cx - 0.08 * h)):int(cx + 0.08 * h) + 1]
		if col.size == 0:
			return None
		col = col.max(axis=1)
		rows = np.flatnonzero(col > 0.5 * (col.max() + background))
		return rows[0] / h if len(rows) else None

	tops = [glyph_top(x) if c.isalnum() else None for c, x in zip(line.txt, line.char_xs)]
	x_tops = [t for c, t in zip(line.txt, tops) if c in X_HEIGHT_CHARS and t is not None]
	cap_tops = [t for c, t in zip(line.txt, tops) if c in CAP_HEIGHT_CHARS and t is not None]
	if not x_tops or not cap_tops:
		return
	x_top, cap_top = float(np.median(x_tops)), float(np.median(cap_tops))
	if x_top - cap_top < 0.08:
		return
	chars = list(line.txt)
	for i, (c, t) in enumerate(zip(chars, tops)):
		if c in 'CcOoSsZz' and t is not None:
			word_start = i == 0 or not chars[i - 1].isalpha()
			if abs(t - x_top) < abs(t - cap_top):
				chars[i] = c.lower()
			elif word_start:
				# capitals never appear inside a word
				chars[i] = c.upper()
	line.txt = ''.join(chars)

# The detector sometimes splits one text line into two overlapping boxes, such
# as "When entering Stealth, c" + "Create a". Replace such pairs with one box
# covering both, so the line is recognized as a whole. Icons detected as small
# separate boxes next to the text are left alone.
def merge_row_fragments(boxes):
	merged = []
	for box, score in sorted(boxes, key=lambda item: float(np.min(item[0][:, 0]))):
		x1, x2 = float(np.min(box[:, 0])), float(np.max(box[:, 0]))
		y1, y2 = float(np.min(box[:, 1])), float(np.max(box[:, 1]))
		for i, (mbox, mscore) in enumerate(merged):
			mx1, mx2 = float(np.min(mbox[:, 0])), float(np.max(mbox[:, 0]))
			my1, my2 = float(np.min(mbox[:, 1])), float(np.max(mbox[:, 1]))
			h = min(y2 - y1, my2 - my1)
			# the left box must be text, not an icon (the right one can be a
			# smaller fragment, e.g. next to underlined text)
			if h <= 0 or h < 0.5 * max(y2 - y1, my2 - my1) or mx2 - mx1 < 2 * (my2 - my1):
				continue
			if min(y2, my2) - max(y1, my1) >= 0.6 * h and x1 > (mx1 + mx2) / 2.0 and x1 - mx2 < 0.5 * h:
				nx1, ny1, nx2, ny2 = mx1, min(y1, my1), max(x2, mx2), max(y2, my2)
				merged[i] = (np.array([[nx1, ny1], [nx2, ny1], [nx2, ny2], [nx1, ny2]], dtype=box.dtype), min(score, mscore))
				break
		else:
			merged.append((box, score))
	return merged

# use substitution to fix common errors in recognized text
def lines_join(separator, iterable):
	result = separator.join(iterable)

	result = re.sub(r'(?<=[a-z])(?:\s+-|-\s+)(?=[a-z])', '-', result, flags=re.I) # [a -b]: extra space (but keep 'a - b')
	result = re.sub(r'^0\+', '+', result) # 0+: symbol recognized as 0
	result = re.sub('%([a-z])', r'% \1', result, flags=re.I) # %a: missing space
	result = re.sub(r'\bt0\b', 'to', result, flags=re.I) # common OCR: t0 -> to
	result = re.sub(r'\bIess\b', 'less', result, flags=re.I) # common OCR: capital I -> l
	result = re.sub(r'(?<=[a-z])0|\b0(?=[a-z]{2})', 'o', result) # 0ver: o recognized as 0
	result = re.sub(r'(?<=\d),(\d%)', r'.\1', result) # 21,4%: decimal point recognized as comma
	result = re.sub(r'(?<=[a-z]) +([:,])(?= |$)', r'\1', result) # Imprinted : extra space
	result = re.sub(r'(?<=[a-z])[CKOSVWXZ](?=[a-z])', lambda m: m.group(0).lower(), result) # PulveriZe: letters that only differ in size (but keep PocQue)
	result = re.sub(r'([,:])(?=[a-z])', r'\1 ', result, flags=re.I) # a,b / a:b: missing space
	result = re.sub(r'(?<=[a-z%])\.(?=[A-Z][a-z]|\+\d)', '. ', result) # end.Start / 1%.This / end.+20: missing space (the game itself sometimes omits it)
	result = re.sub(r'(?<=[a-z]) (s|ve|re|ll|t)\b', r"'\1", result) # Druid s / they ve: missing apostrophe
	result = re.sub(r'(?<=[a-z]{2})(?=\d)', ' ', result, flags=re.I) # with126%: missing space
	result = re.sub(r'\]([a-z])', r'] \1', result, flags=re.I) # ]A: missing space
	result = re.sub(r'\[([\d\.]+)[\s-]+([\d\.]+)\]%', r'[\1 - \2]%', result) # [15.0 -20.0]%: missing space
	result = re.sub(r'([\d,]+)(?:\s+-\s+|\s+-|-\s+)([\d,]+)', r'\1 - \2', result) # [800 -1,000]: missing space

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
	import tkinter as tk
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

	def saveJson():
		filename_without_ext = os.path.splitext(dnav.current_image_filename)[0]
		json_path = os.path.join(dnav.dir_path, filename_without_ext + '.json')
		print(f'Saving: \'{json_path}\'')

		jsonstr = T.get(1.0, tk.END)
		with open(json_path, 'w') as outfile:
			print('Writing:')
			print(jsonstr)
			outfile.write(jsonstr)
	
	win = tk.Tk()
	win.geometry("800x600")
	win.title("D4 Item Data")

	button_frame = tk.Frame(win)
	button_frame.pack(fill=tk.X, side=tk.BOTTOM)

	button_frame_top = tk.Frame(win)
	button_frame_top.pack(fill=tk.X, side=tk.TOP)

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

	status_label = tk.Label(button_frame_top, text="Status")
	save_button = tk.Button(button_frame_top, text="Save JSON", command = saveJson)

	button_frame_top.columnconfigure(0, weight=4)
	button_frame_top.columnconfigure(1, weight=1)

	status_label.grid(row=0, column=0, sticky=tk.W+tk.E)
	save_button.grid(row=0, column=1, sticky=tk.W+tk.E)

	win.after(0, showImage)
	tk.mainloop()

# run tests in directory (checks all images that have a definition (.json))
def runTestsInDir(dir_path):
	dnav = SimpleDirectoryNavigator(dir_path, run_tests_mode=True)
	it_ocr = D4ItemTooltipOCR()

	print("THIS FEATURE IS NOT FULLY IMPLEMENTED")

	while (t := dnav.getImagePath_RunTestsMode(1)) is not None:
		path, has_prev, has_next = t
		print(f"Test path: {path}")

	print("")


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--source-img', type=str, default='examples/screenshot_001.png', help='path to the source image')
	parser.add_argument('--json-output', type=str, default=None, help='output path for item tooltip json data')
	parser.add_argument('--find-tooltip', default=True, type=lambda x: x.lower() not in ['false', 'no', '0', 'None'], help='toggle find tooltip in source [true/false]')
	parser.add_argument('--debug', default=False, type=lambda x: x.lower() not in ['false', 'no', '0', 'None'], help='toggle debug mode [true/false]')
	parser.add_argument('--rec-model-dir', type=str, default='paddleocr-models/en_PP-OCRv3_rec-d4_tooltip', help='legacy D4 recognition model directory')
	parser.add_argument('--det-model-name', type=str, default='PP-OCRv4_mobile_det', help='PaddleOCR 3.x text detection model name')
	parser.add_argument('--run-tests-dir', type=str, default=None, help='run tests in directory')
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
	
	# run tests mode
	elif opt.run_tests_dir != None:
		runTestsInDir(opt.run_tests_dir)
		cv2.destroyAllWindows()

	# output json data from source
	else:
		try:
			it_ocr = D4ItemTooltipOCR(opt.rec_model_dir, opt.det_model_name)
		except Exception as exc:
			print(f'ERROR: Failed to initialize OCR: {exc}', file=sys.stderr)
			sys.exit(1)
		jsonstr = it_ocr.processImage(opt.source_img, opt.find_tooltip, debug=False)

		if jsonstr is None:
			# Keep stdout clean for machine-readable output and make command-line
			# failures visible to the shell/CI via a non-zero status.
			sys.exit(1)

		if (opt.json_output):
			with open(opt.json_output, 'w') as outfile:
				outfile.write(jsonstr)
			print(f'Item tooltip json written to \'{opt.json_output}\'.', end='\n\n')
		else:
			print(jsonstr)