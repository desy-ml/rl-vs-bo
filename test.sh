ks1=(-0.03 -0.02 -0.01 0.0 0.01 0.02 0.03)
ks2=(-0.03 -0.02 -0.01 0.0 0.01 0.02 0.03)
Ks=()
for p1 in $ks1;do;for p2 in $ks2;do;Ks+="$p1 $p2";done;done
echo $Ks
