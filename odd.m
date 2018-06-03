function x=odd(x,updown)

if nargin<2
  updown=1;
end
  
if updown==1
  %returns smallest odd integer value that is greater than or equal to x
  x=ceil(x);
  modindex=find(mod(x,2)==0);
  x(modindex)=x(modindex)+1;
else
  %returns smallest odd integer value that is less than or equal to x
  x=floor(x);
  modindex=find(mod(x,2)==0);
  x(modindex)=x(modindex)-1;
end