import os
config_dir = os.path.join(os.path.dirname(__file__), 'configs', 'remote_eval_configs')
assert os.path.exists(config_dir), f"Config directory {config_dir} does not exist"

if __name__ == '__main__':
    eval_config_path = os.path.join(os.path.dirname(__file__), 'configs', 'remote_eval_config.yml')
    CMD = []

    for dirpath, dirnames, filenames in os.walk(config_dir):
        # print(dirpath, dirnames, filenames)
        if len(dirnames) == 0:
            for filename in filenames:
                if not filename.endswith('.yml'):
                    continue
                o = os.path.join(dirpath, filename)
                assert os.path.exists(o), f"File {o} does not exist"
                # print(o)
                # run(o)
                CMD.append(f'sudo rm -f {eval_config_path}')
                CMD.append(f'sudo cp "{o}" {eval_config_path}')
                log_dir = o.replace('.yml', '.log')
                CMD.append(f'bash eval_remote.sh 2>&1 | sudo tee "{log_dir}"')
                CMD.append(f'sleep 5\n')

    with open(os.path.join(os.path.dirname(__file__), 'ALL.sh'),'w') as f:
        f.write('\n'.join(CMD))