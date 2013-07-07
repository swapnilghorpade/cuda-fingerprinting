
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
	bool **field;
	int rows = 31;
	int columns = 25;
	field = (bool**) malloc (rows * sizeof(bool*));
	for (int i = 0; i < rows; i++) 
		field[i] = (bool*) calloc (columns, sizeof(bool));

	FieldFilling(field,rows,columns,Hull,NHull);
	for (int i = 0; i< rows; i++) {
		for (int j = 0; j < columns ; j++) 
			printf("%d ",field[i][j]);
		printf("\n");
	}
	free(arr);
	free(Hull);
	for (int i = 0;i<rows;i++)
		free(field[i]);
	system("PAUSE");
	free(field);
    return 0;
}
