from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
from PIL import Image
import os

# 모델과 프로세서 로드
processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl")

root = "/workspace/hdd0/byeongcheol/Data/GTA5_patch_car/sty_pos_var_image"
image_list = os.listdir(root)
image_list = sorted(image_list)

images = []
for img_path in image_list:

    img_path = os.path.join(root, img_path)
    image = Image.open(img_path)
    images.append(image)

text_input = ["Describe what you see in the picture."] * len(images)

inputs = processor(images=images, text=text_input, return_tensors="pt")

output_ids = model.generate(**inputs)

for i, (img_path, output_id) in enumerate(zip(image_list, output_ids)):
    generated_text = processor.decode(output_id, skip_special_tokens=True)
    print(f"Generated Text for Image {i+1} - {img_path}: {generated_text}")





# # 이미지 불러오기
# image = Image.open("/workspace/hdd0/byeongcheol/Data/GTA5_patch_car/images/18_Dystopian Noir style car.png")

# # 텍스트 입력 예제
# text_input = "Describe what you see in the picture."

# # 이미지 및 텍스트 입력을 처리하여 모델에 입력
# inputs = processor(images=image, text=text_input, return_tensors="pt")

# # 결과 생성
# output_ids = model.generate(**inputs)

# # 생성된 텍스트 디코딩 및 출력
# generated_text = processor.decode(output_ids[0], skip_special_tokens=True)
# print("Generated Text:", generated_text)
