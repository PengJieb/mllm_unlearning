'''
Author: PengJie pengjieb@mail.ustc.edu.cn
Date: 2026-03-09 17:44:31
LastEditors: PengJie pengjieb@mail.ustc.edu.cn
LastEditTime: 2026-03-09 17:42:48
FilePath: /mllm_unlearning/test.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import lmstudio as lms

model = lms.llm("dataset/Qwen3.5-9B-Aggresive")
result = model.respond("What is the meaning of life?")

print(result)