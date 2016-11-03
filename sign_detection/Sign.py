class Sign(object):
    category = property()
    name = property()
    icon = property()

    def __init__(self, category, name, icon):
        self._category = category
        self._name = name
        self._icon = icon

    @category.setter
    def category(self, value):
        self._category = value

    @name.setter
    def name(self, value):
        self._name = value

    @icon.setter
    def icon(self, value):
        self._icon = value
