from .base_data_element import BaseDataElement


class MultiTaskDataSample(BaseDataElement):

    @property
    def tasks(self):
        return self._data_fields
