# Transfer workspace memory regression

`test_memory_modules.c` exercises repeated CLASS initialization and cleanup
with synthetic parameters. Level 9 includes CMB transfer, lensing, and spectral
distortions. It prints a transfer/spectrum fingerprint before cleanup and the
process RSS after cleanup. This test does not need MontePython or cluster access.

The original `transfer_workspace_free()` released the seven child arrays but
left `ptw` allocated. On the local ARM64 build, `transfer_workspace_init()`
allocated 631 workspaces of 224 bytes per evaluation. The macOS `leaks` tool
found 2,524 retained workspaces (565,376 bytes) after four evaluations, all
rooted at that function. Adding `free(ptw)` after the child frees reduced the
reported leak count to zero after both four and 40 evaluations. The original
build retained 25,240 workspaces (5,653,760 bytes) after 40 evaluations,
confirming linear growth; the corrected build retained zero.

The before/after numerical fingerprint was identical:

```text
q_size=631
q10=1.4768695299531158e-05
transfer_sample=-4.7461994011610206e-07
sd_x_size=500
sd_di10=1.4957748067008855e-13
```

For a fresh local check on macOS, build an isolated copy to avoid replacing an
existing CLASS or `classy` binary:

```sh
cd /path/to/ExoCLASS_perso
CLASS_SOURCE_DIR=$(pwd)
TEST_BUILD_DIR=$(mktemp -d /tmp/exoclass-transfer-test.XXXXXX)
cp -R source include tools main external Makefile "$TEST_BUILD_DIR/"
cd "$TEST_BUILD_DIR"
make -j2 CC=clang 'CPP=clang++ --std=c++11 -fpermissive -Wno-write-strings' \
  'OMPFLAG=' CLASSDIR="$CLASS_SOURCE_DIR" libclass.a
clang -g -O1 -Iinclude -Iexternal/heating -Iexternal/HyRec2020 \
  -Iexternal/RecfastCLASS -c "$CLASS_SOURCE_DIR/test/test_memory_modules.c" \
  -o memory_modules.o
clang++ -g memory_modules.o libclass.a -o memory_modules
./memory_modules 9 40
MallocStackLogging=1 leaks --groupByType --nostacks --atExit -- \
  ./memory_modules 9 40
```

RSS may still rise during allocator warm-up even when `leaks` finds zero
retained allocations. The test uses a synthetic standard-model setup with
`l_max_scalars=300`; it does not certify the production DarkAges and PCA1000
paths, error exits, or six-rank MPI. The corrected source needs a fresh build,
finite Planck+FOSSIL preflight, and memory-growth test under the exact operator
configuration before any production resubmission.
