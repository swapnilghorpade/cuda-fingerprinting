
#include "BuildConvexHull.h"
#include "FieldFilling.h"
int main()
{
	int N;
	scanf("%d",&N);
	Point *arr;
	arr = (Point*) malloc (N * sizeof(Point));
	for (int i = 0 ; i < N; i++ ) {
		int x,y;
		scanf("%d",&x);
		scanf("%d",&y);
		Point *temp = new Point(x,y);
		arr[i] = *temp;
	}
	Point *Hull;
	Hull = (Point*) malloc (N*sizeof(Point));
	int NHull = 0;
	Build(arr,N,Hull,&NHull);
	for (int i = 0 ; i < NHull; i ++ )
		Hull[i].print();
	system("PAUSE");
	bool *field;
	int rows = 31;
	int columns = 25;
	field = (bool*) malloc ((rows * columns) * sizeof(bool));
	FieldFilling(field,rows,columns,Hull,NHull);
	for (int i = 0; i< rows; i++) {
		for (int j = 0; j < columns ; j++) 
			printf("%d ",field[i*columns+j]);
		printf("\n");
	}
	free(arr);
	free(Hull);
	system("PAUSE");
	free(field);
    return 0;
}
