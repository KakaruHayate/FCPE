import numpy as np

def split_into_sentences(arr, min_silence_duration=500):
    """
    将一维的静音/有声音序列切分成句子。
    
    参数：
    arr : numpy数组，元素为bool类型，True表示静音，False表示有声音。
    min_silence_duration : 最小静音持续时间（毫秒），默认为500毫秒。
    
    返回：
    sentences : 列表，每个元素为元组(start_frame, end_frame)，表示句子的起始和结束帧。
    """
    if len(arr) == 0:
        return []
    
    # 转换最小静音持续时间到帧数
    min_silence_frames = min_silence_duration // 10
    
    # 找到状态变化的索引
    transitions = np.where(arr[:-1] != arr[1:])[0] + 1
    segments = []
    prev = 0
    current_state = arr[0]
    for pos in transitions:
        segments.append((prev, pos - 1, current_state))
        current_state = not current_state
        prev = pos
    # 添加最后一个段
    segments.append((prev, len(arr) - 1, current_state))
    
    # 筛选出足够长的静音段
    long_mute_segments = [s for s in segments if s[2] and (s[1] - s[0] + 1) >= min_silence_frames]
    
    # 确定分割区间
    intervals = []
    if not long_mute_segments:
        # 没有长静音段，检查是否存在非静音段
        non_mute = [s for s in segments if not s[2]]
        if non_mute:
            interval_start = 0
            interval_end = len(arr) - 1
            intervals.append((interval_start, interval_end))
    else:
        # 处理第一个区间
        first_mute_start = long_mute_segments[0][0]
        if first_mute_start > 0:
            intervals.append((0, first_mute_start - 1))
        
        # 处理中间的区间
        for i in range(len(long_mute_segments) - 1):
            current_mute = long_mute_segments[i]
            next_mute = long_mute_segments[i + 1]
            interval_start = current_mute[1] + 1
            interval_end = next_mute[0] - 1
            if interval_start <= interval_end:
                intervals.append((interval_start, interval_end))
        
        # 处理最后一个区间
        last_mute = long_mute_segments[-1]
        if last_mute[1] < len(arr) - 1:
            intervals.append((last_mute[1] + 1, len(arr) - 1))
    
    # 生成句子
    sentences = []
    for interval in intervals:
        interval_start, interval_end = interval
        # 收集该区间内的所有非静音段
        relevant_segments = [s for s in segments if not s[2] and s[0] >= interval_start and s[1] <= interval_end]
        if relevant_segments:
            # 取所有非静音段的最小起始和最大结束
            sentence_start = min(s[0] for s in relevant_segments)
            sentence_end = max(s[1] for s in relevant_segments)
            sentences.append((sentence_start, sentence_end))
    
    return sentences

# 示例使用
if __name__ == "__main__":
    # 示例数组：假设有声音（False）和静音（True）交替出现
    # 每帧代表10毫秒，这里创建一个示例数组
    arr = np.array([False]*10 + [True]*50 + [False]*20 + [True]*30 + [False]*10 + [True]*30, dtype=bool)
    #arr=np.load(r"E:\AUFSe04BPyProgram/AUFSd04BPyProgram/fcbe/fcope/op/train/mask/1/2025-02-04_22-43-56.npy")
    sentences = split_into_sentences(arr, 300)
    print("句子边界（帧）：", sentences)
    # 转换为时间（毫秒）
    sentences_ms = [(start*10, (end+1)*10) for start, end in sentences]
    print("句子边界（毫秒）：", sentences_ms)