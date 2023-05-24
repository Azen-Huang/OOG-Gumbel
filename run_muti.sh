for ((i = 82; i <= 200; i = i + 1))
do
    echo "iter: $i Start"
    startTime_s=$(date +%s)
    
    g++ -march=native Gumbel.cc -g -ltensorflow -fopenmp -O2 -o Gumbel && "./"Gumbel --self_play $((i)) 1 &
    sleep 5 && g++ -march=native Gumbel.cc -g -ltensorflow -fopenmp -O2 -o Gumbel && "./"Gumbel --self_play $((i)) 2 &
    sleep 10 && g++ -march=native Gumbel.cc -g -ltensorflow -fopenmp -O2 -o Gumbel && "./"Gumbel --self_play $((i)) 3 &  
    sleep 15 && g++ -march=native Gumbel.cc -g -ltensorflow -fopenmp -O2 -o Gumbel && "./"Gumbel --self_play $((i)) 4 &  
    sleep 20 && g++ -march=native Gumbel.cc -g -ltensorflow -fopenmp -O2 -o Gumbel && "./"Gumbel --self_play $((i)) 5 &
    sleep 25 && g++ -march=native Gumbel.cc -g -ltensorflow -fopenmp -O2 -o Gumbel && "./"Gumbel --self_play $((i)) 6 &
    sleep 30 && g++ -march=native Gumbel.cc -g -ltensorflow -fopenmp -O2 -o Gumbel && "./"Gumbel --self_play $((i)) 7 &  
    sleep 35 && g++ -march=native Gumbel.cc -g -ltensorflow -fopenmp -O2 -o Gumbel && "./"Gumbel --self_play $((i)) 8 &  
    wait
    
    endTime_s=$(date +%s)
    sumTime=$[$endTime_s - $startTime_s]
    echo "Self play cost time: $sumTime"
    
    python -u "./train_network.py" $((i))

    if ((i % 10 == 0 ))
    then
        #g++ -march=native KataGo.cc -g -ltensorflow -fopenmp -O2 -o KataGo && "./"KataGo --evaluation $i | tee output_$((i))_vs_$((i - 10)).log
        g++ -march=native Gumbel.cc -g -ltensorflow -fopenmp -O2 -o Gumbel && "./"Gumbel --evaluation $i 0 | tee output_$((i))_vs_$((i - 10)).log
        python -u "./copy_folder.py" -1
    fi
done
