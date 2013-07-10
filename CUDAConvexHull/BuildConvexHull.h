#include "Point.h"
Point FirstPoint = Point(0,0);

int compare(const void * a, const void * b) {
    int result = 1;
	Point V1 = *(Point*) a;
	Point V2 = *(Point*) b;
    if ((V1.Subtract(FirstPoint).VectorProduct(V2.Subtract(FirstPoint)) < 0) ||
        ((V1.Subtract(FirstPoint).VectorProduct(V2.Subtract(FirstPoint)) == 0) &&
		(V1.Distance(FirstPoint) < V2.Distance(FirstPoint))))
        result = -1;
	if (V1.Equals(V2))
        result = 0;
    return result;
}

void Sorting(Point *arr, int N) {
	for (int i = 0 ; i < N; i++) {
		if (arr[i].Y > (FirstPoint).Y)
            (FirstPoint) = arr[i];
        if ((arr[i].Y == (FirstPoint).Y) && (arr[i].X < (FirstPoint).X))
            (FirstPoint) = arr[i];
    }
	qsort(arr, N, sizeof(Point), compare);
}

void BuildHull(int *arr, int N,int *IntHull,int *NHull) {
	Point *Minutiae;
	Minutiae = (Point*) malloc(N * sizeof(Point));
	for (int i = 0; i < N; i++)
		Minutiae[i] = Point(arr[2*i],arr[2*i+1]);
	Sorting(Minutiae,N);
	Point *Hull = (Point*) malloc (N *sizeof(Point));
	Hull[0] = Minutiae[0];
	Hull[1] = Minutiae[1];
	int top = 1;
	int nextToTop = 0;
    for (int i = 2; i < N; i++)    {
		while ((Minutiae[i].Subtract(Hull[nextToTop]).VectorProduct(Hull[top].Subtract(Hull[nextToTop])) <=0) && (!Hull[top].Equals(FirstPoint))) {
			 top--;
			 if (Hull[top].Equals(FirstPoint))
                 nextToTop = top;
             else
                 nextToTop = top-1;
         }
         top++;
		 nextToTop = top-1;
		 Hull[top] = Minutiae[i];
    }
	*NHull = top+1;
	for (int i =0 ; i < *NHull; i++) {
		IntHull[2*i] = Hull[i].X;
		IntHull[2*i+1] = Hull[i].Y;
	}
	free(Minutiae);
	free(Hull);
}
