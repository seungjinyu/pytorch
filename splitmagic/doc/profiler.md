1. A forward 

### test_node_a_<MODEL>.py

### run_node_a()

### client.request_template_plan() 
* template_plan_path = /tmp/jin_template_plan_a.tsv

row_id      실행순서 기준 고유 번호  
op          연산 종류 
idx         해당 op 의 occurence 번호 
suffix      어떤 tensor 인지 
shape       tensor shape


``` 
[ex]
12    relu    3    result    [128,64,32,32]
12번째 기록된 tensor 연산 종류는 relu, local index 는 3 , 그 relu 에서 기록한 tensor 종류는 result, shape 는 [128,64,32,32]
```

2. A payload 


3. A -> B

4. B Backward

5. B opimizer / state reply
