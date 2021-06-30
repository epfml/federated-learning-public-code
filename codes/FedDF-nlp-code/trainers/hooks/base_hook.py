import inspect


def build_args(func, **kwargs):
    """select parameters from **kwargs according to input to func"""
    specs = inspect.getfullargspec(func)
    if specs.varkw is not None or specs.varargs is not None:
        print(func)
        print(specs)
        raise ValueError("I'm not expecting *args, **kwargs of `func` ...")
    needed_args = set(specs.args)
    defaults = []
    if specs.defaults is not None:
        defaults = [arg for arg in specs.defaults]
    start_idx = len(specs.args) - len(defaults)
    assert len(specs.args[start_idx:]) == len(defaults)
    output = {name: default for name, default in zip(specs.args[start_idx:], defaults)}
    # override defaults if necessary
    output.update({name: val for name, val in kwargs.items() if name in needed_args})
    return output


class Hook(object):
    def __init__(self):
        self._trainer = None
        self._disabled = False

    @property
    def trainer(self):
        return self._trainer

    @property
    def conf(self):
        return self._trainer.conf

    @property
    def disabled(self):
        return self._disabled

    @property
    def model(self):
        return self._trainer.model

    @property
    def batch_step(self):
        return self._trainer.batch_step

    @property
    def epoch_step(self):
        return self._trainer.epoch_step

    @property
    def log_fn(self):
        return self._trainer.log_fn

    @property
    def log_fn_json(self):
        return self._trainer.log_fn_json

    def on_train_begin(self, **kwargs):
        pass

    def on_batch_end(self, **kwargs):
        pass

    def on_train_end(self, **kwargs):
        pass

    def on_validation_end(self, **kwargs):
        pass


def _patch(func):
    def _call(container, **kwargs):
        returns = []
        for hook in container.hooks:  # e.g. recorder
            if hook.disabled:
                continue
            hook_func = getattr(hook, func.__name__)  # recorder.on_train_end
            _args = build_args(hook_func, **kwargs)  # sort out **kwards
            returns.append(hook_func(**_args))  # recorder.on_train_end(args)
        return returns

    return _call


class HookContainer(Hook):
    def __init__(self, world_env, hooks=None):
        super(HookContainer, self).__init__()
        self._world_env = world_env
        self.hooks = []
        if hooks:
            self.hooks = self._init_hooks(hooks)

    def _init_hooks(self, hooks):
        for attr_name, attr_val in self._world_env.items():
            for hook in hooks:
                setattr(hook, "_" + attr_name, attr_val)
                # i.e., hook._trainer = world_env[trainer]
                # e.g. recorder._trainer = world_env[trainer]
        return hooks

    @_patch
    def on_train_begin(self, **kwargs):
        pass

    @_patch
    def on_batch_end(self, **kwargs):
        pass

    @_patch
    def on_train_end(self, **kwargs):
        pass

    @_patch
    def on_validation_end(self, **kwargs):
        pass
