function [elements, counts] = parse_formula(formula)
%PARSE_FORMULA Split a chemical formula into element symbols and counts.
%   [elements, counts] = PARSE_FORMULA('AgNbLaAlO6') returns
%   elements = {'Ag','Nb','La','Al','O'}, counts = [1 1 1 1 6].
%
%   Uses regexp to match an uppercase letter + optional lowercase letter,
%   followed by an optional integer count (default 1).

tokens = regexp(formula, '([A-Z][a-z]?)(\d*)', 'tokens');
n = numel(tokens);
elements = cell(1, n);
counts = zeros(1, n);
for k = 1:n
    elements{k} = tokens{k}{1};
    if isempty(tokens{k}{2})
        counts(k) = 1;
    else
        counts(k) = str2double(tokens{k}{2});
    end
end
end
