from modelscope.pipelines import pipeline
sv_pipeline = pipeline(
    task='speaker-verification',
    model='/data02/datasets/funasr/CAM',
    model_revision='v1.0.0'
)
speaker1_a_wav = 'https://modelscope.cn/api/v1/models/damo/speech_campplus_sv_zh-cn_16k-common/repo?Revision=master&FilePath=examples/speaker1_a_cn_16k.wav'
speaker1_b_wav = 'https://modelscope.cn/api/v1/models/damo/speech_campplus_sv_zh-cn_16k-common/repo?Revision=master&FilePath=examples/speaker1_b_cn_16k.wav'
speaker2_a_wav = 'https://modelscope.cn/api/v1/models/damo/speech_campplus_sv_zh-cn_16k-common/repo?Revision=master&FilePath=examples/speaker2_a_cn_16k.wav'
# 相同说话人语音
result = sv_pipeline([speaker1_a_wav, speaker1_b_wav])
print(result)
# 不同说话人语音
result = sv_pipeline([speaker1_a_wav, speaker2_a_wav])
print(result)
# 可以自定义得分阈值来进行识别，阈值越高，判定为同一人的条件越严格
# result = sv_pipeline([speaker1_a_wav, speaker2_a_wav], thr=0.31)
# print(result)

# 可以传入output_emb参数，输出结果中就会包含提取到的说话人embedding
# result = sv_pipeline([speaker1_a_wav, speaker2_a_wav], output_emb=True)
# print(result['embs'], result['outputs'])
# # 可以传入save_dir参数，提取到的说话人embedding会存储在save_dir目录中
# result = sv_pipeline([speaker1_a_wav, speaker2_a_wav], save_dir='savePath/')