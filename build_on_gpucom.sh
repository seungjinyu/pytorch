# 2️⃣ CUDA 아키텍처 제한 (네 GPU에 맞게)
export TORCH_CUDA_ARCH_LIST="8.6"

# 3️⃣ 병렬 빌드 설정
export MAX_JOBS=$(nproc)

# to prevent oom 
# export MAX_JOBS=$(( $(nproc) - 2 ))

# 4️⃣ GCC 12 강제 (CUDA 호환)
export CC=gcc-12
export CXX=g++-12
export CUDAHOSTCXX=g++-12

# 5️⃣ (선택) 테스트 빌드 제거
export BUILD_TEST=0

# 6️⃣ 클린 (컴파일러 바꿨다면 필수)
# python3 setup.py clean
# rm -rf build
# rm -rf third_party/nccl/nccl/build

# 7️⃣ 🔥 최종 빌드
"${CONDA_PREFIX}/bin/python3" setup.py develop
