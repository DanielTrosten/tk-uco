import pickle
from pydantic import BaseModel


class Config(BaseModel):
    _glob_vars = tuple()

    class Config:
        validate_all = True
        extra = "forbid"

    @property
    def class_name(self):
        return self.__class__.__name__

    def set_globs(self, globs=None):
        if globs is None:
            globs = {}
        else:
            for key, value in globs.items():
                if hasattr(self, key):
                    setattr(self, key, value)

        for k in self._glob_vars:
            globs[k] = getattr(self, k)

        for _, value in self:
            if isinstance(value, Config):
                value.set_globs(globs=globs)

    def to_pickle(self, file_name):
        with open(file_name, "wb") as f:
            pickle.dump(self, f)
