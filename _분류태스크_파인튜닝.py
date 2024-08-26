from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
import evaluate
import numpy as np
import torch

# 모델 및 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained('eenzeenee/t5-base-korean-summarization')
model = AutoModelForSeq2SeqLM.from_pretrained('eenzeenee/t5-base-korean-summarization')

# 데이터셋 로딩
train_dataset = load_dataset('csv',
                             data_files=r"C:\Users\DESKTOP\Desktop\work\KoBART-summarization\data\train_final.tsv",
                             split='train', delimiter='\t')
test_dataset = load_dataset('csv',
                            data_files=r'C:\Users\DESKTOP\Desktop\work\KoBART-summarization\data\test.tsv',
                            split='train', delimiter='\t')

dataset = DatasetDict({
    'train': train_dataset,
    'test': test_dataset
})

# 슬라이딩 윈도우 함수 정의
def sliding_window(text, max_len, overlap):
    start_points = range(0, len(text), max_len - overlap)
    return [text[start:start + max_len] for start in start_points]

# 토큰화 함수 정의a
def tokenize_function(examples):
    prefix = "summarize: "
    tokenized_inputs = []
    tokenized_labels = []

    # 각 예제를 처리
    for dialogue, summary in zip(examples['dialogue'], examples['summary']):
        chunks = sliding_window(dialogue, max_len=512, overlap=100)
        for chunk in chunks:
            inputs = prefix + chunk
            model_input = tokenizer(inputs, max_length=512, truncation=True, padding="max_length", return_tensors="pt")
            label = tokenizer(summary, max_length=256, truncation=True, padding="max_length", return_tensors="pt")
            
            tokenized_inputs.append(model_input)
            tokenized_labels.append(label['input_ids'])
    
    # 텐서를 하나의 텐서로 병합
    inputs_tensor = {key: torch.cat([x[key] for x in tokenized_inputs], dim=0) for key in tokenized_inputs[0]}
    labels_tensor = torch.cat(tokenized_labels, dim=0)
    
    inputs_tensor['labels'] = labels_tensor
    return inputs_tensor

# 데이터셋에 토큰화 함수 적용
tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=['dialogue', 'summary'])

# 데이터 콜레이터 설정
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# 훈련 설정
training_args = Seq2SeqTrainingArguments(
    output_dir=r"C:\Users\DESKTOP\Desktop\work\KoBART-summarization\results",
    evaluation_strategy="epoch",  # 에폭마다 평가
    learning_rate=5.6e-5,  # 더 안정적인 학습을 위한 조금 더 낮은 학습률
    per_device_train_batch_size=8,  # 증가된 배치 크기
    per_device_eval_batch_size=8,  # 훈련 배치 크기와 일관된 평가 배치 크기 유지
    weight_decay=0.01,  # 정규화를 위해 weight decay 계속 사용
    save_strategy="epoch",  # 에폭마다 모델 저장
    save_total_limit=5,  # 디스크 공간을 절약하기 위해 체크포인트 총 수 제한
    num_train_epochs=5,  # 총 훈련 에폭 수
    predict_with_generate=True,  # 예측을 위해 generate 사용
    report_to="none",  # 외부 서비스로의 보고 비활성화
    gradient_accumulation_steps=8,  # 실제 배치 크기와 메모리에 따라 조정
    lr_scheduler_type="linear",  # 선형 스케줄러
    warmup_ratio=0.1  # 훈련 초기 10% 동안 warmup,
)

# 평가 메트릭 설정
rouge_metric = evaluate.load("rouge")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    result = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels)
    return {key: np.mean(value) * 100 for key, value in result.items()}

# 트레이너 설정
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)
trainer.train()

# 훈련된 모델 및 토크나이저 저장
trainer.save_model()
tokenizer.save_pretrained(training_args.output_dir)