from pathlib import Path
import numpy as np
import os
from katacr.build_dataset.utils.split_part import process_part
import cv2
import katacr.build_dataset.constant as const
from katacr.build_dataset.utils.split_part import extract_bbox
from cnocr import CnOcr
from katacr.build_dataset.constant import part3_elixir_params
import re


class OCR_version0:
    def __init__(self):
        self.ocr = CnOcr(det_model_name='naive_det')

    def __call__(self, img):
        return self.ocr.ocr(img)
    
    def process_part1(self, img_time, show=True):
        results=self(img_time)
        if show:
            print("OCR results:", results)
            cv2.imshow('time', img_time)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        stage=None
        print(results)
        if len(results) < 2:
           return np.inf

        text_detection=results[0]['text']
        print(text_detection)
        

        if '时间' in text_detection:
            stage=0
        if '加时赛' in text_detection:
            stage=1
        print(stage)

        
        time_detection=results[1]['text']
        
        match = re.search(r'(\d):(\d{2})', time_detection)
        if match:
            try:
                m = int(match.group(1))
                s = int(match.group(2))
            except ValueError:
                return np.inf
        else:
            return np.inf
        if stage is None or m is None or s is None: return np.inf
        t = m * 60 + s
        if stage == 0:
            return 180 - t
        return 180 + 120 - t
    
    
    
    def process_part3_elixir(self, img_part3):
        from katacr.build_dataset.constant import part3_elixir_params
        img = extract_bbox(img_part3, *part3_elixir_params)
        # cv2.imwrite(f"/home/yy/Coding/datasets/Clash-Royale-Dataset/images/part3_elixir_classification/{self.save_count:3}.jpg", img)
        # self.save_count += 1  # DEBUG: elixir position
        results = self(img)
        try:
            m = int(results[0]['text'])
            if m > 10:
                m = m % 10  # wrong detection
        except ValueError:
                m = None
        if results[0]['text'] == 'O' or results[0]['text'] == 'o':
            m=0
        return m
    
## There are 2 versions to use, decide later..


class OCR:
    def __init__(self):
        self.ocr = CnOcr(det_model_name='naive_det')

    def __call__(self, img):
        return self.ocr.ocr(img)
    
    def process_part1(self, img_time_stage0, img_time_stage1, show=False):
        try:
            results=self(img_time_stage0)
            if show:
                cv2.imshow('time', img_time_stage0)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            stage=None
        
            if len(results) ==1:
                stage=1
                results = self(img_time_stage1)
                if show:
                        cv2.imshow('time', img_time_stage1)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
                time_detection=results[0]['text']
                match = re.search(r'(\d):(\d{2})', time_detection)
                if match:
                    try:
                        m = int(match.group(1))
                        s = int(match.group(2))
                    except ValueError:
                        return np.inf
                else:
                    return np.inf
                if stage is None or m is None or s is None: return np.inf
                t = m * 60 + s
                return 180 + 120 - t

            text_detection=results[0]['text']
        

            if '时间' in text_detection:
                stage=0


        
            time_detection=results[1]['text']
            
            match = re.search(r'(\d):(\d{2})', time_detection)
            if match:
                try:
                    m = int(match.group(1))
                    s = int(match.group(2))
                except ValueError:
                    return np.inf
            else:
                return np.inf
            if stage is None or m is None or s is None: return np.inf
            t = m * 60 + s
            if stage == 0:
                return 180 - t
            elif stage == 1:
                return 180 + 120 - t
            else:
                return np.inf
        except Exception as e:
            return np.inf  # ⬅️ global fallback in case of any unexpected error

    
    
    
    def process_part3_elixir(self, img_part3):
        from katacr.build_dataset.constant import part3_elixir_params
        img = extract_bbox(img_part3, *part3_elixir_params)
        # cv2.imwrite(f"/home/yy/Coding/datasets/Clash-Royale-Dataset/images/part3_elixir_classification/{self.save_count:3}.jpg", img)
        # self.save_count += 1  # DEBUG: elixir position
        results = self(img)
        try:
            m = int(results[0]['text'])
            if m > 10:
                m = m % 10  # wrong detection
        except ValueError:
                m = None
        if results[0]['text'] == 'O' or results[0]['text'] == 'o':
            m=0
        if m is not None:
            m = abs(m)  # ensure positive
        return m