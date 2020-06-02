

import read_restart_file
def test_readpic():
    filename = "examples/pyretis.restart"
    info = read_restart_file(filename)   
    print(info)

