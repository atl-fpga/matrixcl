#
#  USAGE:
#     make          ... to build the programs
#     make clean    ... to remove object and executable files
#

# verify that you are including the right make.def file for the platform
include make.def

EXES     = matmul$(EXE)

MMULOBJS = host.$(OBJ)

all: $(EXES)

matmul$(EXE): $(MMULOBJS) 
	$(CLINKER) $(CFLAGS) $(OPENCLFLAGS) -o $@ $^ $(LIBS)

host.$(OBJ): matvec_mul.cl

clean:
	$(RM) $(EXES) *.$(OBJ)

veryclean:
	$(RM) $(EXES) *.$(OBJ)

.SUFFIXES:
.SUFFIXES: .c .cpp .$(OBJ)

.c.$(OBJ):
	$(CC) $(CFLAGS) -c $<

.cpp.$(OBJ):
	$(CC) $(CFLAGS) -c $<


