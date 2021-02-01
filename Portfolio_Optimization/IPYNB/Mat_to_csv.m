%converts the FF48.mat to a cvs file (data.cvs)
load('FF48.mat');
writematrix( IndustryPortfolios2, 'Data.csv' );
type 'Data.csv'