# [ CV Fall Project ]
# ImmerVision - Real Time 3D HoloProjection

**Ji Hun Seo (서지훈, 2021-19397)** markseo0424@snu.ac.kr \
**Moon Seok Khang (강문석, 2022-15470)** mskhang@snu.ac.kr \
**Na Rim Kim (김나림, 2022-17320)** wingedlz@snu.ac.kr \
**Woo Hyun Kim (김우현, 2022-13625)** tryyoung@snu.ac.kr

## 개요

가상 3D환경을 별도의 장치 없이 3차원으로 인식할 수 있게끔, 운동시차를 활용해 착시 이미지를 Real-time으로 출력한다. 더 나아가, 실제 환경의 Lighting을 반영한 가상 3차원 환경의 실시간 렌더링을 목표로 하여, 더욱 현실감 있는 3차원 뷰를 구현하는 것이 목표이다.

## 사용 라이브러리
아래 명령어를 통해 설치. (pytorch는 별도 command line 있음.)
```commandline
pip install -r requirements.txt
```

**opencv** - webcam의 접근, 간단한 영상처리, 영상 출력.

- 참고 - https://076923.github.io/posts/Python-opencv-2/

**numpy** - matrix multiplication 등에 활용.

**mediapipe** - 얼굴 인식 등 built-in function 테스트용, 여건이 안될 시 활용 가능.

**torch** - 딥러닝용 라이브러리로, matrix의 GPU연산을 쉽게 가능하게 해준다.\
설치 코드 (터미널에 복사) : 
```commandline
 pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
 ```

## 프로젝트 구성

### 인식부
1. 웹캠으로 사용자 얼굴 인식
2. 인식된 값을 바탕으로 관찰자의 3차원 위치 계산
### 표현부
3. 계산된 값을 바탕으로 3D Projection (wireframe rendering)
4. 카메라의 위치 및 lighting으로 렌더링 (shaded rendering)
