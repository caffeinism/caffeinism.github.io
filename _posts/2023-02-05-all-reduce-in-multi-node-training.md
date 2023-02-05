---
title: "NCCL all-reduce와 multi-node training"
excerpt: "multi-node training에서의 고려사항"

categories:
  - Infrastructure
  - ML
tags:
  - [nccl, ml, pytorch]

permalink: /posts/2023-02-05-all-reduce-in-multi-node-training/

toc: true
toc_sticky: true

date: 2023-02-05
last_modified_at: 2023-02-05
---

# 들어가며

많은 회사에서 컴퓨팅 자원이 쌓이면서 여러 노드에서 학습하려는 니즈가 더 늘고 있는 것 같습니다. 하지만 실제로 학습을 진행하고 속도 수치화를 해보면 성능이 큰 폭으로 늘어나지는 않는 것을 발견하게 됩니다.

큰 기업에서 일해본 적이 없기 때문에 그사세를 알 수는 없겠지만 중소규모의 회사나 학교에서 사용하는 컴퓨팅 인프라의 네트워크 대역폭은 크게 높지 않은 경우가 많습니다. 필요하다고 느낀 누군가가 요청하지 않으면 1Gbps 스위치에 모든 노드를 묶어서 사용하는 경우도 많습니다. 한 노드 안에서 학습할 때는 당연히 큰 문제를 느끼지 못하지만 여러 노드로 확장하면서 발목을 잡히는 경우가 많습니다. 엔비디아는 DGX를 출시하면서 Mellanox사의 Infiniband adapter를 탑재하다 못해 2020년에 Mellanox사를 인수하고 네트워크 시장의 파이를 먹고 있습니다. 그만큼 딥러닝 인프라에도 네트워크는 중요하다고 생각합니다.

본론으로 돌아와서 우리가 PyTorch에서 Distributed Data Parallel(DDP)이 어떻게 학습을 진행하는지 대략 알아보고 네트워크 오버헤드가 있을 수 있는 부분을 탐색해보고자 합니다.

# Distributed Data Parallel

## 어떻게 학습되는지?

![1](/assets/images/posts_img/2023-02-05-all-reduce-in-multi-node-training/1.svg)

기본적으로 DDP로 학습을 진행하려면 모든 노드가 모든 데이터에 접근할 수 있어야 합니다.

![2](/assets/images/posts_img/2023-02-05-all-reduce-in-multi-node-training/2.svg)

이후 epoch이 시작할 때 모든 노드가 다른 데이터를 가지고 있도록 동등하게 분배합니다. PyTorch의 경우 DistributedSampler가 지정된 노드 수와 내가 몇 번째 노드인지 등을 보고 데이터를 선택하는 역할을 합니다.

![3](/assets/images/posts_img/2023-02-05-all-reduce-in-multi-node-training/3.svg)
이제 iteration이 시작되고 forward-backward를 진행합니다. 현재 각자 다른 data로 구한 gradient가 계산 되었을 것입니다. 이걸 그대로 step 한다면 노드마다 다른 데이터로 학습된 각기 다른 모델이 되어버립니다.

![4](/assets/images/posts_img/2023-02-05-all-reduce-in-multi-node-training/4.svg)
모든 gradient의 합을 구하고

![5](/assets/images/posts_img/2023-02-05-all-reduce-in-multi-node-training/5.svg)
각자 나눠 가져야 합니다 (all-reduce). 합쳐진 gradient를 이용해 step 한다면 한 iteration 학습 완료! 우리는 이제 여러 노드에 걸쳐서 하나의 큰 배치를 학습할 수 있습니다.

## 근데 느려요

여러 노드를 이용해 학습해보면 투입한 GPU에 비해 별로 안 빨라지거나 두 배나 투입했는데 심지어 더 느려진 속도를 얻는 경우도 있습니다. 왜 그럴까요?

모델의 파라미터 수를 $$S$$라고 하고 파라미터를 모으고자 하는 네트워크 링크 대역폭 중에 가장 좁은 대역폭이 $$B$$라고 한다면 아무리 최적화를 잘한다고 해도 한번 내보내는데 $$\frac{S}{B}$$만큼의 시간이 걸리고 다 더한 값을 받는데 $$\frac{S}{B}$$의 시간이 걸려 총 $$\frac{2S}{B}$$의 시간이 소요됨을 알 수 있습니다. (정확히는 $$\frac{S}{B} * \frac{2(n-1)}{n}$$의 시간이 걸립니다. all-reduce 알고리즘들에 대해서는 [^1][^2]를 참고하면 좋습니다. GPU가 많아지면 $$\frac{2(n-1)}{n}$$는 결국 2와 크게 다를 바 없기 때문에 간단하게 두 배라고 치겠습니다.)

특정 모델로 예시를 들어보자면 요즘 transformer를 기반으로 한 모델들은 파라미터 수가 엄청나게 많습니다. ViT를 예시로 들어보면 가장 작은 ViT-Base가 86M의 파라미터를 가지고 있고 ViT-Huge는 632M인데 [paperswithcode](https://paperswithcode.com/sota/image-classification-on-imagenet)에 들어가 보면 요즘엔 이것보다도 큰 모델이 엄청 많죠.

또한 일반적으로 작은 규모에서 많이 사용하는 10Gbps 네트워크를 가정해서 ViT-Base를 학습한다고 생각해봅시다. 86M의 파라미터는 half precision의 적용을 가정했을 때 172MB입니다. $$\frac{2S}{B}$$는 0.275초입니다. 한번 모든 노드가 all-reduce 하는 데 0.275초가 걸린다는 거죠. 얼마 안 걸리는 것 같지만 여기서 중요한 건 제 GPU가 한 iteration을 학습하는 데는 0.4초밖에 걸리지 않는다는 겁니다. GPU는 0.4초만 일하는데 학습한 결과를 모두 더하는 데만 0.275초가 걸려버리는 거죠.

노드당 GPU가 8개씩 있는 머신이 세 대가 있다고 가정해봅시다. 한 노드 안에서 8 GPU로 DDP 학습을 시킨다면 대충 한 iteration에 0.42초 정도가 걸릴 것입니다. 하지만 세 노드에 각각 GPU가 8대씩 있어도 10Gbps 이더넷 스위치로 묶여있다면 한 iteration에 0.675초가 걸리게 됩니다. 그렇게 되면 GPU를 8개에서 24개로 늘렸음에도 불구하고 성능은 고작 1.86배밖에 향상되지 않는 결과를 얻게 됩니다.

# 결론

DGX는 이런 문제를 줄일 수 있는 최적의 구성을 위해 NVLink, Infiniband adapter 등을 탑재하고 있습니다. NVLink는 한 노드 내에서 학습할 때의 all-reduce의 오버헤드를 줄여주지만 이미 PCIe의 대역폭은 나누어 쓰는 것을 가정해도 엄청나게 느리진 않기 때문에 위의 예제의 8 GPU 학습에서 0.0x 초를 줄여줄 뿐이고 multi-node training 시에는 결국 가장 느린 링크가 병목이 되기 때문에 큰 도움이 되지 않습니다. (물론 네트워크가 아주 높은 대역폭을 가지게 구성되었다면 PCIe 스위치가 병목이 될 수 있습니다) 결국 multi-node training을 성공적으로 하기 위해서는 높은 대역폭을 가진 네트워크 구성이 필수적입니다.

[^1]: [https://images.nvidia.com/events/sc15/pdfs/NCCL-Woolley.pdf](https://images.nvidia.com/events/sc15/pdfs/NCCL-Woolley.pdf)
[^2]: [https://developer.nvidia.com/blog/massively-scale-deep-learning-training-nccl-2-4/](https://developer.nvidia.com/blog/massively-scale-deep-learning-training-nccl-2-4/)