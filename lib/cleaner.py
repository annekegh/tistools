import os

def cleaner(fn_i,fn_o,overwrite=False):
    """remove everything until the last load (ld) path, and remove the last ld path too

    fn_i  --  filename input
    fn_o  --  filename output
    fn_i+".TEMPORARYFILE.TMP"  --  name for temporary file
    """

    if fn_o == fn_i:
        assert overwrite
    if overwrite:
        assert fn_i == fn_o
        fn_o = fn_i+".TEMPORARYFILE.TMP"   # name for temporary file
        # move back at the end
    if os.path.exists("fn_o"):
        assert overwrite

    with open(fn_i, 'r') as fi:
        lines = fi.readlines()
    
    fo = open(fn_o, 'w')
    
    printer = False
    for line in lines:
        if 'ld' in line:
            printer = False
            # This erases all previous
            fo.close()
            fo = open(fn_o, 'w')
        elif 'ACC' in line and 'sh' in line:
            printer = True
        if printer:
            fo.write(line)

    fi.close()
    fo.close()

    if overwrite:
        os.remove(fn_i)
        os.rename(fn_o, fn_i)
        print("overwrite file...",fn_i,"(was temporarily %s)"%fn_o)
    else:
        print("file written...",fn_o)

