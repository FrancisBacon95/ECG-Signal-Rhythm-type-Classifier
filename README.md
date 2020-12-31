# mitbih-arrhythmia dataset을 이용한 비트(Superclass), 리듬 타입 정리(Deep Learning 활용)


	2020 산학협력 "MIT-BIH ArrhyThmia Dataset을 활용한 심장질환 분류모델"

## 2) 4가지 Rhythm Type Classifier 개발

분석 모델 : RandomForest & Decision Tree in Python using Colab

## 4가지 리듬별 특성 정리 - ML_feature.pdf

* (B
	- 'V' beat 비율 0.4 ~ 0.5
	- 'N' beat 비율 0.3 ~ 0.5
* (N
	- 'N' beat 비율 0.6 ~ 0.9
* (SVTA
	- 'A' beat 비율 0.75 ~ 0.95
* (VT
	- 'N' beat 비율 0.7 ~ 0.8과 'V' beat 비율 0.15 ~ 0.25 or 'V' beat 비율 0.75
	
### 적합 모델 결정 과정
* 2D-CNN 적용 - 최소한으로 특징을 파악할 수 있는 이미지로 용량을 낮추었지만 여전히 메모리 용량 및 컴퓨터 성능 부족 문제가 발생하였다.
* 1D-CNN 적용 - 논문에서 확인한 window size = 2700 적용 시 데이터 개수가 합쳐서 10000개에도 미치지 않으며, 이마저도 '(N' rhythm이 대부분이라 학습이 제대로 되지 않는 문제가 발생하였다. 임의로 줄여서 window size = 200까지 줄여서 해보았으나 마찬가지로 데이터 개수 부족으로 학습이 어렵다고 판단하였다.
* 목적이 '(N', '(B', '(SVTA', '(VT'에 해당하는 Feature를 찾는 것이기 때문에 Random Forest나 Decision Tree가 적합하다고 판단하였다.

### 2-1) Rhythm Type 중 "(N", "(B", "(SVTA", "(VT"만 가져와서 각각 분류 - 03_Rhythm_Type_Classifier_ML.ipynb

목적 :  4가지 Rhythm Type 증상에 대한 특성 분석 + 특성에 맞는 feature 추출, 분류에 적합한 모델 적용 및 개발

ML 적용 이유 : 각 리듭 타입별 존재하는 샘플 갯수가 1개에서 530개까지 존재하며, DL을 적용시키기엔 각 리듬별 패턴 파악을 위한 샘플 갯수가 충분하지 않다고 판단

#### preprocessing)

* 원 데이터로부터, 각 리듬별 'count'(=한 리듬타입 안에 찍힌 심박수 값의 총 개수), 'R_count', 'R_mean', 'R_count_per_beat'(=R_cnt/beat개수), 'R_count_per_sig'(=R_cnt/count)에 대한 추가 정보와 각 리듬타입이 존재하는 비율('N', 'R', '[', ... etc)에 대한 데이터 생성

* 전체 비트 타입과 리듬 내의 최댓값을 통해서 R peak를 계산(max 값의 0.9이상을 R로 판단)하였고 이를 feature로 넣어서 Random Forest와 Decision Tree를 적용하였다.

/* (시행착오)

* 해석 용이성을 위해 변수 차원을 줄이기 위한 PCA, t-SNE 적용 ==> 새로운 변수들에 대한 설명력들이 부족 ==> 적용 X

*/

* Random Forest Model 1 )


* Result) Mean Accuracy = 97.1%, Mean F1 score = 97.1%, 
	Confusion matrix = 
	- [[ 51   4   0   0]  - (B
	-  [  0 133   0   0]  - (N
	-  [  0   1   5   1]  - (SVTA
	-  [  0   0   0  15]] - (VT



* Decision Tree 2 )


* Result) Mean Accuracy = 94.8%, Mean F1 score = 94.9%, 
	Confusion matrix = 
	- [[ 48   7   0   0]  - (B
	-  [  1 132   0   0]  - (N
	-  [  0   2   5   0]  - (SVTA
	-  [  0   1   0  14]] - (VT
 
 
### 2-2) Rhythm Type 전체에서 "(N", "(B", "(SVTA", "(VT" 각각 분류 - 04_Rhythm_Type_Classifier_ML.ipynb

* 원 데이터로부터, 각 리듬별 'count'(=한 리듬타입 안에 찍힌 심박수 값의 총 개수), 'R_count', 'R_mean', 'R_count_per_beat'(=R_cnt/beat개수), 'R_count_per_sig'(=R_cnt/count)에 대한 추가 정보와 각 리듬타입이 존재하는 비율('N', 'R', '[', ... etc)에 대한 데이터 생성

* 전체 비트 타입과 리듬 내의 최댓값을 통해서 R peak를 계산(max 값의 0.9이상을 R로 판단)하였고 이를 feature로 넣어서 Random Forest와 Decision Tree를 적용하였다.


* ['(VFL', '(SBR', '(IVR', '(BII', '(AB'] 의 Train/Test 분류 부정확한 것 제외한 모델.

* Random Forest Model 1 )


* Result) 전체 Rhythm에 대한 Mean Accuracy = 81.4%, Mean F1 score = 80.9%, 
	Confusion matrix = 
	- [[ 14   4   0  14   0   0   0   0   0   0]  - (AFIB
 	-  [  1  10   0   3   0   0   0   0   0   0]  - (AFL
 	-  [  0   0  64   2   0   0   0   0   0   0]  - (B
 	-  [  9   0   1 133   0   0  14   0   3   0]  - (N
 	-  [  0   0   0   1  10   0   0   0   0   0]  - (NOD
 	-  [  0   0   0   0   0  18   0   0   0   0]  - (P
 	-  [  0   1   0  17   0   0  13   0   0   0]  - (PREX
 	-  [  0   0   0   1   0   0   0   7   0   0]  - (SVTA
 	-  [  0   0   0   0   0   0   0   0  25   0]  - (T
 	-  [  0   0   0   1   0   0   0   0   0  17]] - (VT

+ Accuracy of '(N' : 0.831
+ Accuracy of '(B' : 0.970
+ Accuracy of '(SVTA' : 0.875
+ Accuracy of '(VT' : 0.944
(소수 넷째자리 반올림)

* Decision Tree 2 )
	


* Result) 전체 Rhythm에 대한 Mean Accuracy = 76.7%, Mean F1 score = 69.8%, 
	Confusion matrix = 
	- [[  0   0   0  32   0   0   0   0   0   0]  - (AFIB
	-  [  0   0   0  14   0   0   0   0   0   0]  - (AFL
	-  [  0   0  64   1   0   0   0   0   1   0]  - (B
	-  [  0   0   0 155   0   0   0   0   4   0]  - (N
	-  [  0   0   0   0  11   0   0   0   0   0]  - (NOD
	-  [  0   0   0   0   0  18   0   0   0   0]  - (P
	-  [  0   0   0  31   0   0   0   0   0   0]  - (PREX
	-  [  0   0   0   1   0   0   0   7   0   0]  - (SVTA
	-  [  0   0   0   4   0   0   0   0  21   0]  - (T
	-  [  0   0   0   1   0   0   0   0   0  17]] - (VT

+ Accuracy of '(N' : 0.969
+ Accuracy of '(B' : 0.970
+ Accuracy of '(SVTA' : 0.875
+ Accuracy of '(VT' : 0.944
(소수 넷째자리 반올림)


## 결론
- 4 가지 rhythm type인 '(N', '(B', '(SVTA', '(VT' 증상에 대한

1) 특성 분석
* (B
	- 'V' beat 비율 0.4 ~ 0.5
	- 'N' beat 비율 0.3 ~ 0.5
* (N
	- 'N' beat 비율 0.6 ~ 0.9
* (SVTA
	- 'A' beat 비율 0.75 ~ 0.95
* (VT
	- 'N' beat 비율 0.7 ~ 0.8과 'V' beat 비율 0.15 ~ 0.25 or 'V' beat 비율 0.75

2) 특성에 맞는 Feature 추출
- 4 가지의 rhythm type을 분류할 때 우선적으로 rhythm내의 'V' beat의 비율, 그 다음으로는 'N' beat의 비율이 random forest의 변수 중요도 그래프를 통해서 파악할 수 있었다.

- 임의로 정한 R값이 'V', 'N' beat에 이어 중요한 Feature인 것을 확인할 수 있었다.

3) 분류에 적합한 모델 적용 및 개발
- Random Forest와 Decision Tree를 이용하여 분류 모델 적용 및 Feature Importance를 파악하였다.
