function [elements, counts] = parse_formula(formula)
%PARSE_FORMULA Split a chemical formula into element symbols and counts.
%   [elements, counts] = PARSE_FORMULA('AgNbLaAlO6') returns
%   elements = {'Ag','Nb','La','Al','O'}, counts = [1 1 1 1 6].
%
%   Uses regexp to match an uppercase letter + optional lowercase letter,
%   followed by an optional integer count (default 1).

[tokens, ~] = regexp(formula, '([A-Z][a-z]?)(\d*)', 'tokens', 'match');
n = numel(tokens);
elements = cell(1, n);
counts = zeros(1, n);
consumed = '';
for k = 1:n
    elements{k} = tokens{k}{1};
    if isempty(tokens{k}{2})
        counts(k) = 1;
    else
        counts(k) = str2double(tokens{k}{2});
    end
    consumed = [consumed, tokens{k}{1}, tokens{k}{2}]; %#ok<AGROW>
end

% The regex silently skips characters it can't match (e.g. '(', '-').
% Verify the whole formula was consumed so bad input fails loudly here
% rather than producing wrong features downstream.
if ~strcmp(consumed, formula)
    error('parse_formula:badFormula', ...
        'Could not fully parse formula ''%s''.', formula);
end
end
