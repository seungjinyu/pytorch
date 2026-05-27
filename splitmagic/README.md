# 현재 목표

네 연구의 핵심:

```text
Node A:
forward only

Node B:
backward only
```

그런데 backward는 원래 forward 때 저장된 tensor(saved tensor)가 필요함.

그래서:

```text
Node A가 saved tensor를 수집해서 보내고
Node B가 그걸 사용해 backward 수행
```

이 구조를 만들고 있는 중.

---

# 지금까지 만든 구조

05.18.26

## 프로젝트 구조

```text
splitmagic/
├── splitmagic/
│   ├── __init__.py
│   ├── runtime.py
│   ├── inspector.py
│   ├── hooks.py
│   └── payload.py
│
├── test_import.py
├── test_inspector.py
└── test_payload.py
```

---

# 각 파일 역할

## 1. runtime.py

아직 껍데기 단계.

```python
SplitRuntime(model, role="A")
SplitRuntime(model, role="B")
```

이런 형태의 framework API 시작점.

즉:

```text
사용자는 splitmagic만 import
```

하게 만들기 위한 입구.

---

## 2. hooks.py

PyTorch module hook 관리.

역할:

```text
Conv2d
ReLU
MaxPool
Linear
...
```

같은 layer들의:

```text
input shape
output shape
module 이름
```

을 기록.

예:

```text
conv1 (Conv2d):
(4,3,32,32)
→
(4,8,32,32)
```

---

## 3. inspector.py

핵심.

여기서 사용한 기능:

```python
torch.autograd.graph.saved_tensors_hooks
```

이걸 통해:

```text
PyTorch가 backward를 위해 실제 저장한 tensor
```

를 자동 추적.

즉:

```text
Conv backward가 뭘 저장하는지
BN backward가 뭘 저장하는지
MaxPool이 indices를 저장하는지
```

를 직접 보기 시작함.

---

# 지금 가능한 것

현재는:

```python
report = inspect_saved_tensors(...)
```

하면:

## A. Saved tensor 목록

```text
[0] shape=(...)
[1] shape=(...)
...
```

## B. Module trace

```text
conv1
relu
pool
fc
...
```

를 얻을 수 있음.

즉:

```text
이 모델에서 backward를 위해 어떤 tensor가 필요한지
```

를 자동 조사 가능해짐.

---

# 4. payload.py

Node A가 전송할 데이터 구조.

현재 기능:

```python
payload.add_tensor(...)
payload.save(...)
payload.load(...)
```

즉:

```text
saved tensor들을 파일 형태로 저장 가능
```

---

# 현재 전체 흐름

지금 구현된 흐름은:

```text
model forward
↓
saved_tensors_hooks가
PyTorch 내부 saved tensor 감지
↓
inspector가 기록
↓
payload로 저장
↓
payload.pt 생성
```

---

# 아직 안 한 것

아직 중요한 것들은 남아 있음.

## 아직 없음

```text
❌ Node B replay
❌ saved tensor overwrite
❌ autograd graph 재구성
❌ ZeroMQ 전송
❌ gradient matching
```

---

# 지금 단계의 의미

사실 지금 단계가 매우 중요함.

왜냐면 네 연구의 첫 번째 난제가:

```text
"대체 어떤 tensor를 보내야 하는가?"
```

였는데,

이제는:

```text
PyTorch가 실제 저장하는 tensor를 자동으로 조사 가능
```

한 상태가 됐기 때문.

즉:

```text
payload schema 자동 추출의 시작점
```

까지 온 거야.

---

# 다음 단계 후보

다음부터는 갈림길이 있음.

## 방향 1

saved tensor와 module trace 연결

```text
saved_tensor:0
→ conv1.input

saved_tensor:1
→ conv1.weight
```

같은 semantic naming.

---

## 방향 2

payload를 실제 replay 가능한 형태로 구성.

---

## 방향 3

Node B dummy forward 실험 시작.

---

지금은 사실:

```text
PyTorch가 backward를 위해 뭘 저장하는지
관찰 가능한 framework
```

까지 만든 상태라고 보면 돼.
