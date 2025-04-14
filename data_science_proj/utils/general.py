import yaml


def load_yaml(argument):
    return yaml.load(open(argument), Loader=yaml.FullLoader)

def parse_args(all_args):

    tags = ['-','--']
    all_args = all_args[1:]

    results = {}
    key = None

    for arg in all_args:
        if any(tag in arg for tag in tags):
            if key is not None:
                results[key] = True
            key = arg.lstrip('-').lower()
        else:
            if key is not None:
                results[key] = arg
                key = None
    if key is not None:
        results[key] = True
    return results

def load_config(sys_args):

    args = parse_args(sys_args)
    params = load_yaml(args['config'])
    params['cl'] = args

    return params
