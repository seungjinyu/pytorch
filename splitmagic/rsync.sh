rsync -avh \
  --exclude='.git/' \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='*.pt' \
  --exclude='*.log' \
  --exclude='experiment_logs/' \
  --exclude='experiments/' \
  --exclude='build/' \
  --exclude='dist/' \
  --exclude='*.egg-info/' \
  --exclude='nsys_results/' \
  --exclude='nsys_results_remote/' \
  --exclude='merged_profiles/' \
  --exclude='menus_offline/' \
  /home/syu23/seungjin/pytorch/splitmagic/ \
  ubuntu@10.32.126.244:/home/ubuntu/splitmagic_project/