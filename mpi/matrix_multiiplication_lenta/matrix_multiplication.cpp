/*
данный алгоритм реализует ленточное перемножение матриц с использванием библиотеки MPI (A*B = C)
*/

#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <random>

#define MASTER 0               /* id первого потока*/
#define FROM_MAIN 1      
#define FROM_SHADOW 2        
#define NumberRowsA 72                
#define NumberColumbsA 10              
#define NumberColumbsB 15              


int main(int argc, char *argv[])
{
	double	a[NumberRowsA][NumberColumbsA],          
		b[NumberColumbsA][NumberColumbsB],           
		c[NumberRowsA][NumberColumbsB];
	int	numtasks,              // количесвто потоков доступных сейчас
		taskid,                // id потока  
		numworkers,            // число worker 
		averow,				//среднее число строк для отправки
		extra,
		offset,			
		source,                //  id сообщения потока для приему
		dest,                  // id сообщения для отправки 
		stype,                 // текущий тип отправки сообщений (в workers  или в главный поток) 
		rows,                  // число строк для отправки в каждый worker 
		i, j, k, rc;          
	    
	MPI_Status status;
	//подготовка
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
	MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
	numworkers = numtasks - 1;


	//иницилизация матриц и отправка в главном потоке
	if (taskid == MASTER)
	{
		printf("mpi start wit %d tasks.\n", numtasks);
		printf("Initializing matrix A\n");
		std::default_random_engine gemetrator;
		std::uniform_real_distribution<double> distribution(0,250);
		for (i = 0; i < NumberRowsA; i++)
		{
			printf("\n");
			for (j = 0; j < NumberColumbsA; j++) {
				a[i][j] = i + j;
				printf("%8.2f   ", a[i][j]);
			}
		}

		printf("\n******************************************************\n");
		printf("Initializing matrix B\n");
		for (i = 0; i < NumberColumbsA; i++)
		{
			printf("\n");
			for (j = 0; j < NumberColumbsB; j++) {
				b[i][j] = i * j;
				printf("%8.2f   ", b[i][j]);
			}
		}
		//  отправляем матрицы в worker  потоки


		//вычисляем размеры лент
		averow = NumberRowsA / numworkers;
		extra = NumberRowsA % numworkers;
		offset = 0;

		stype = FROM_MAIN;
		for (dest = 1; dest <= numworkers; dest++)
		{
			if (dest <= extra) { rows = averow + 1; }
			else { rows = averow; }
			printf("\n");
			printf("Sending %d rows to task %d offset=%d\n", rows, dest, offset);
			MPI_Send(&offset, 1, MPI_INT, dest, stype, MPI_COMM_WORLD);
			MPI_Send(&rows, 1, MPI_INT, dest, stype, MPI_COMM_WORLD);
			MPI_Send(&a[offset][0], rows*NumberColumbsA, MPI_DOUBLE, dest, stype,
				MPI_COMM_WORLD);
			MPI_Send(&b, NumberColumbsA*NumberColumbsB, MPI_DOUBLE, dest, stype, MPI_COMM_WORLD);
			offset = offset + rows;
		}

		// получаем результат из workers
		stype = FROM_SHADOW;
		for (i = 1; i <= numworkers; i++)
		{
			source = i;
			MPI_Recv(&offset, 1, MPI_INT, source, stype, MPI_COMM_WORLD, &status);
			MPI_Recv(&rows, 1, MPI_INT, source, stype, MPI_COMM_WORLD, &status);
			MPI_Recv(&c[offset][0], rows*NumberColumbsB, MPI_DOUBLE, source, stype,
				MPI_COMM_WORLD, &status);
		}

		printf("******************************************************\n");
		printf("Result Matrix:\n");
		for (i = 0; i < NumberRowsA; i++)
		{
			printf("\n");
			for (j = 0; j < NumberColumbsB; j++)
				printf("%8.2f   ", c[i][j]);
		}
	}


	// код выполняющийчя в workers
	if (taskid > MASTER)
	{
		stype = FROM_MAIN;
		MPI_Recv(&offset, 1, MPI_INT, MASTER, stype, MPI_COMM_WORLD, &status);
		MPI_Recv(&rows, 1, MPI_INT, MASTER, stype, MPI_COMM_WORLD, &status);
		MPI_Recv(&a, rows*NumberColumbsA, MPI_DOUBLE, MASTER, stype, MPI_COMM_WORLD, &status);
		MPI_Recv(&b, NumberColumbsA*NumberColumbsB, MPI_DOUBLE, MASTER, stype, MPI_COMM_WORLD, &status);
		// алгоритм ленточного перемножения матрицы
		for (k = 0; k < NumberColumbsB; k++)
			for (i = 0; i < rows; i++)
			{
				c[i][k] = 0.0;
				for (j = 0; j < NumberColumbsA; j++)
					c[i][k] = c[i][k] + a[i][j] * b[j][k];
			}
		stype = FROM_SHADOW;
		MPI_Send(&offset, 1, MPI_INT, MASTER, stype, MPI_COMM_WORLD);
		MPI_Send(&rows, 1, MPI_INT, MASTER, stype, MPI_COMM_WORLD);
		MPI_Send(&c, rows*NumberColumbsB, MPI_DOUBLE, MASTER, stype, MPI_COMM_WORLD);
	}
	MPI_Finalize();
}