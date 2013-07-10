#include "BuildConvexHull.h"
#include "FieldFilling.h"
#include "WorkingArea.h"

int main()
{
	/*int N;
	printf("Enter amount of Minutiae: ");
	scanf("%d",&N);
	int *arr;
	arr = (int*) malloc (2*N * sizeof(int));
	printf("Enter Minutiae:\n");
	for (int i = 0 ; i < N; i++ ) {
		scanf("%d",&arr[2*i]);
		scanf("%d",&arr[2*i+1]);
	}
	int *Hull;
	Hull = (int*) malloc (2*N*sizeof(int));
	int NHull = 0;
	BuildHull(arr,N,Hull,&NHull);
	printf("ConvexHull:\n");
	for (int i = 0 ; i < NHull; i ++ )
		printf("%d %d\n",Hull[2*i],Hull[2*i+1]);
	bool *field;
	printf("Enter rows and columns:\n");
	int rows,columns;
	scanf("%d",&rows);
	scanf("%d",&columns);
	field = (bool*) malloc ((rows * columns) * sizeof(bool));
	FieldFilling(field,rows,columns,Hull,NHull);
	printf("FieldFilling:\n");
	for (int i = 0; i< rows; i++) {
		for (int j = 0; j < columns ; j++) 
			printf("%d ",field[i*columns+j]);
		printf("\n");
	}
	printf("Enter radius:\n");
	int radius;
	scanf("%d",&radius);
	BuildWorkingArea(field,rows,columns,radius);
	printf("Working Area:\n");
	for (int i = 0; i< rows; i++) {
		for (int j = 0; j < columns ; j++) 
			printf("%d ",field[i*columns+j]);
		printf("\n");
	}
	free(arr);
	free(Hull);
	free(field);
	system("PAUSE");*/
    return 0;
}
