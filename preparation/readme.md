# 데이터 전처리

한국어를 문장 단위로 녹음한 영상을 입부분 모양과 음성으로 분할하는 작업입니다. 학습 시 자막 레이블은 주어졌다고 가정합니다. 

---
## setup

필요  dependency-packages를 설치합니다.
~~~
pip install -r requirements.txt
~~~


### Generate wav file

영상에서 음성을 추출하는 작업입니다. 
~~~
python generate_wav.py --test_dir [My_dir]
~~~

1. test_dir : 영상의 위치를 기입합니다.
   * 음성은  test_dir 이하 wav 라는 폴더에 생성됩니다. 

### Generate mouth ROIs video

영상에서 입 모양 부분을 추출하여 비디오로 나타냅니다. 

#### 원본 영상 

<img src='https://github.com/alcien/avsr_test/blob/main/asset/lip_K_5_M_04_C955_A_012_9.gif' style='width:200px'>
</img>

#### 입 모양 영상

<img src='https://github.com/alcien/avsr_test/blob/main/asset/mouth_lip_K_5_M_04_C955_A_012_9.gif' width:200px height:100px>
</img>

#### 얼굴 landmark를 찾기 위한 필요 파일 다운로드
~~~
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bzip2 -d shape_predictor_68_face_landmarks.dat.bz2

wget --content-disposition https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks/raw/master/preprocessing/20words_mean_face.npy
~~~

#### 입 모양 영상 추출
~~~
python generate_mouth.py --test_dir [dir] --face_predictor_path [face predictor file path] --mean_face_path [mean face file path]
~~~

1. test_dir : 원본 영상이 담긴 폴더
    * 입 모양 영상은 test_dir 이하 mouth 폴더에 저장됩니다.     
2. face_predictor_path : shape_predictor_68_face_landmarks.dat의 위치
3. mean_face_path : 20words_mean_face.npy의 위치



## demo용 파일 만들기


위와 같은 프로세스를 가지나, 파일 하나에 대해서 확인해보고 싶을 때 사용합니다. 

### Generate wav file

영상에서 음성을 추출하는 작업입니다. 
~~~
python generate_wav_infer.py --test_dir [My_dir] --test_fn [filename]
~~~

1. test_dir : 영상이 위치한 폴더
   * 음성은  test_dir 이하 wav 라는 폴더에 생성됩니다.
2. test_fn : test_dir에 위치한 wav로 변환시키고 싶은 파일명

### Generate mouth ROIs video

#### 입 모양 영상 추출
~~~
python generate_mouth_infer.py --test_dir [dir] --test_fn [filename] --face_predictor_path [face predictor file path] --mean_face_path [mean face file path]
~~~

1. test_dir : 원본 영상이 담긴 폴더
    * 입 모양 영상은 test_dir 이하 mouth 폴더에 저장됩니다.
2. test_fn : test_dir에 위치한 변환시키고 싶은 파일명 
3. face_predictor_path : shape_predictor_68_face_landmarks.dat의 위치
4. mean_face_path : 20words_mean_face.npy의 위치


# 모델 입력을 위한 훈련, 테스트용 파일 생성
~~~
python generate_csv.py --dataset [folder] --mouth_fd [mouth_dir] --label_fd [label_dir] --wav_fd [wav_dir] --fn [csv filename] --Lfn [label filename] --spm_path [tokenizer_folder]
~~~

1. dataset : mouth, wav, label이 포함된 폴더.
2. mouth_fd : 입 모양 영상이 포함된 폴더
3. label_fd : 모델 라벨인 영상 자막이 들어있는 폴더
4. wav_fd : 음성 파일이 있는 폴더
5. fn : 훈련, 테스트용 파일명(.csv)
~~~
# 예시
# dataset  mouth_fd video_length(#frames)  tokenized text
/home/aiv-gpu-019/data-small,lip_J_6_F_04_C322_A_001_11.mp4,167,22 115 523 140 82 73 80 81 94 70 82 77 391 82 77 198 285 364 132 239 77 319 108 107 88
~~~
6. Lfn : 영상 자막 합본 파일 (.txt)
7. spm_path : 영상 자막 토크나이저가 포함된 폴더
     * sentencepiece 모델로 구현
     * 사용 방법은 generate_spm 노트북을 참조
     * unigram5000.model, unigram5000_units.txt 형식으로 spm/unigram 폴더에 저장되어 있어야 함

# 모델 입력을 위한 demo용 파일 생성
~~~
python generate_infer.py --dataset [folder] --mouth_fd [mouth_dir] --wav_fd [wav_dir] --fn [csv filename] 
~~~

1. dataset : mouth, wav가 포함된 폴더.
2. mouth_fd : 입 모양 영상이 포함된 폴더
3. wav_fd : 음성 파일이 있는 폴더
4. fn : 훈련, 테스트용 파일명(.csv)


  


