# 데이터 전처리 사전 준비
* preparation 폴더의 dependencies-packages 설치 
* 자세한 사항은 prepration 폴더의 README 참고
~~~
cd prepration

pip install -r requirements.txt
cd ..
~~~

# 모델 학습, 테스트

## dependencies-packages 설치
~~~
pip install pip==22.2
pip install -r requirements.txt
git submodule init
git submodule update

cd fairseq
pip install --editable ./ #10분에서 15분 정도 소요
cd ..
pip install --upgrade omegaconf
pip install --upgrade hydra-core
~~~


## 사전 테스트용 모델 파일 위치

* 입 모양 영상으로만 학습한 테스트용 ckpt 파일(video.ckpt, last.ckpt)은 train1207에 저장합니다. 
* 입 모양 영상-음성으로 학습한 ckpt 파일(av.ckpt, last.ckpt)은 t1207_av에 저장합니다.

## 토큰화 알고리즘 세팅 확인
* spm/unigram 폴더의 unigram5000.model, unigram5000.vocab, unigram5000_units.txt 가 bpe1207 폴더 내에 있는 것인지 확인

* 사용자 정의로 모델 라벨을 생성하고 싶을 경우, preparation 폴더의 generation_spm.ipynb 노트북 참고

## 모델 학습

모델의 학습은 scripts 폴더의 train.sh로 진행합니다. 

~~~
bash scripts/train.sh
~~~

### 파일 설명
~~~
model_folder=/DATA/temp/auto_avsr #현재 폴더
train_folder=train1207 # 모델이 있는 폴더, 현재 폴더에 존재해야 함
modality=video #모달리티 종류(입 모양 : video / 입 모양-음성 : audiovisual) 
csv_file=train_aavsr_1207.csv # (학습용 csv 파일, preparation 폴더 참조)
val_file=avsr_1207_1.csv # (valid 용 csv 파일, preparation 폴더 참조) 

mouth=lip2 # 입 모양 영상 폴더 위치 (csv 파일에 기재된 폴더 위치에 영상 폴더가 존재해야 함)
wav=wav2 # 음성 폴더 위치 (csv 파일에 기재된 폴더 위치에 음성 폴더가 존재해야 함)

prepare_model=/DATA/temp/auto_avsr/train1207/last.ckpt (이전에 학습한 모델을 계속에서 학습할 경우, 해당 모델의 학습용 ckpt 위치)

#모델 학습 실행
python $model_folder/train2.py label_flag=1 exp_dir=$model_folder exp_name=$train_folder data.modality=$modality data.dataset.root_dir=/DATA/temp/auto_avsr data.dataset.train_file=$csv_file data.dataset.val_file=$val_file mouth_dir=$mouth wav_dir=$wav trainer.resume_from_checkpoint=$prepare_model
~~~

모델 학습은 train2.py 로 실행

매개변수: 
  * label_flag는 demo 여부를 판단하기 위한 것으로, 학습, 테스트용은 1로 지정합니다.
  * exp_dir, exp_name : 모델이 저장될 위치로, [exp_dir]/[expname]  위치에 학습 폴더 저장
  * data.modality : 데이터 모달리티 종류, video / audiovisual
  * data.dataset.root_dir : 입 모양, 음성 폴더가 위치한 폴더
  * data.dastaset.train_file : 학습용 csv 파일
  * data.dataset.val_file : 테스트용 csv 파일
  * mouth_dir : 입 모양 폴더명
  * wav_dir : 음성 폴더명
  * trainer.resume_from_checkpoint : 학습을 지속할 경우 사용. 학습하던 모델명

    
## 모델 테스트

모델의 테스트는 scripts 폴더의 test.sh로 진행합니다. 

~~~
bash scripts/test.sh
~~~

### 파일 설명
~~~
model_folder=/DATA/temp/auto_avsr #현재 폴더
modality=video  #모달리티 종류(입 모양 : video / 입 모양-음성 : audiovisual)
pretrained_path=/DATA/temp/auto_avsr/train1207/cleaned.ckpt #(테스트용 ckpt 파일, download.py로 받은 av, video.ckpt) 
mouthD=mouth # 입 모양 영상 폴더 위치 (csv 파일에 기재된 폴더 위치에 영상 폴더가 존재해야 함)
wavD=wav # 음성 폴더 위치 (csv 파일에 기재된 폴더 위치에 음성 폴더가 존재해야 함)
csv_file=trainAvsr.csv # (test 용 csv 파일, preparation 폴더 참조) 

#모델 테스트 실행
python $model_folder/eval.py data.modality=$modality data.dataset.root_dir=/DATA/temp/auto_avsr data.dataset.test_file=$csv_file label_flag=1 mouth_dir=$mouthD wav_dir=$wavD pretrained_model_path=$pretrained_path
~~~

모델 테스트는 eval.py 로 실행

매개변수: 
  * label_flag는 demo 여부를 판단하기 위한 것으로, 학습, 테스트용은 1로 지정합니다.
  * data.modality : 데이터 모달리티 종류, video / audiovisual
  * data.dataset.root_dir : 입 모양, 음성 폴더가 위치한 폴더
  * data.dataset.test_file : 테스트용 csv 파일
  * mouth_dir : 입 모양 폴더명
  * wav_dir : 음성 폴더명
  * pretrained_model_path : 테스트를 위해 사용할 모델 ckpt 


## 모델 demo

영상 파일 하나를 테스트해 볼 경우, scripts 폴더의 demo.sh로 진행합니다. 

prepartion 폴더의 과정을 한번에 진행하여, 영상에서 입 모양 및 음성을 추출하여 테스트하는 과정까지를 포함합니다. 

~~~
bash scripts/demo.sh
~~~

### 파일 설명

~~~
mp4_folder=test #테스트할 파일의 폴더 위치
mp4_name=lip_K_5_M_04_C955_A_012_9.mp4 #테스트할 파일명

face_path=shape_predictor_68_face_landmarks.dat #입 모양 추출을 위한 파일
mean_face=20words_mean_face.npy #입 모양 추출을 위한 파일

csv_file=test/demo_test.csv #모델 입력을 위한 csv파일명

python generate_mouth_infer.py --test_dir $mp4_folder --test_fn $mp4_name --face_predictor_path $face_path --mean_face_path $mean_face   #테스트 파일의 입 모양 영상 생성
python generate_wav_infer.py --test_dir $mp4_folder --test_fn $mp4_name #테스트 파일의 음성 추출
python generate_infer.py --dataset $mp4_folder --mouth_fd mouth --wav_fd wav --fn $csv_file #모델 테스트용 demo_test.csv 파일 생성

model_folder=/DATA/temp/auto_avsr #테스트용 모델 위치
modality=video #모달리티 종류(입 모양 : video / 입 모양-음성 : audiovisual)
pretrained_path=/DATA/temp/auto_avsr/train1207/cleaned.ckpt #(테스트용 ckpt 파일, download.py로 받은 av, video.ckpt) 

#파일 테스트 실행
python $model_folder/eval.py data.modality=$modality data.dataset.root_dir=/DATA/temp/auto_avsr data.dataset.test_file=$csv_file label_flag=0 mouth_dir=mouth wav_dir=wav pretrained_model_path=$pretrained_path 

#파일 테스트 결과 확인
cat $model_folder/result.txt
~~~

모델 테스트는 eval.py 로 실행

매개변수: 
 * label_flag는 demo 여부를 판단하기 위한 것으로, demo용은 0로 지정합니다.
 * data.modality : 데이터 모달리티 종류, video / audiovisual
  * data.dataset.root_dir : 입 모양, 음성 폴더가 위치한 폴더
  * data.dataset.test_file : 테스트용 csv 파일
  * mouth_dir : 입 모양 폴더명
  * wav_dir : 음성 폴더명
  * pretrained_model_path : 테스트를 위해 사용할 모델 ckpt 


