version_info = (0, 1, 0)
# format:
# ('smart_major', 'smart_minor', 'smart_patch')


def get_version():
    return "%d.%d.%d" % version_info


__version__ = get_version()
