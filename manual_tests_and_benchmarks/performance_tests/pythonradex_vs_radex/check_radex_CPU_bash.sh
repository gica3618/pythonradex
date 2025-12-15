#this is just to check how much CPU radex uses
#in terminal, do something like /usr/bin/time -v ./check_radex_CPU_bash.sh

for i in $(seq 1 1000); do
    #../../../tests/Radex/bin/radex_static_sphere < radex_test_preformance.inp > /dev/null
    #dev/null seems not to affect performance, and it is safer without, to see any error messages
    ../../../tests/Radex/bin/radex_static_sphere < check_radex_CPU.inp
done