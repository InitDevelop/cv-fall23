# [ CV Fall Project ]
# ImmerVision - Real Time 3D HoloProjection

**Ji Hoon Seo (서지훈, 2021-19397)** markseo0424@snu.ac.kr \
**Moon Seok Khang (강문석, 2022-15470)** mskhang@snu.ac.kr \
**Na Rim Kim (김나림, 2022-17320)** wingedlz@snu.ac.kr \
**Woo Hyun Kim (김우현, 2022-13625)** tryyoung@snu.ac.kr

## 개요

가상 3D환경을 별도의 장치 없이 3차원으로 인식할 수 있게끔, 운동시차를 활용해 착시 이미지를 Real-time으로 출력한다. 더 나아가, 실제 환경의 Lighting을 반영한 가상 3차원 환경의 실시간 렌더링을 목표로 하여, 더욱 현실감 있는 3차원 뷰를 구현하는 것이 목표이다.

## 사용 라이브러리

opencv - webcam의 접근, 간단한 영상처리, 영상 출 및 (여건이 안될 시) 얼굴 인식 기능 사용.

- 참고 - https://076923.github.io/posts/Python-opencv-2/

numpy - matrix multiplication 등에 활용.

## 프로젝트 구성

1. 웹캠으로 사용자 얼굴 인식
2. 인식된 값을 바탕으로 관찰자의 3차원 위치 계산
3. 계산된 값을 바탕으로 3D Projection
4. 카메라의 위치 및 lighting으로 렌더링

