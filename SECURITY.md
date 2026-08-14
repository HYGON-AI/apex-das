# Security Policy

Please do not report security vulnerabilities in a public issue.

Report vulnerabilities through the private security reporting channel
configured for the HYGON-AI organization. Include the affected version,
reproduction steps, impact, and any proposed mitigation.

Do not include production credentials, customer data, private repository URLs,
or other non-public information in reports or test cases.

## Inherited pickle cache

`apex/contrib/sparsity/permutation_search_kernels/exhaustive_search.py` may
read `master_list.pkl` from its current working directory when
`generate_all_unique_combinations` is called and that cache exists. This is an
upstream-inherited, optional performance cache; it is not used during package
import.

Because pickle files can execute code while loading, run this tool only in a
trusted, isolated workspace and remove any cache obtained from an untrusted
source. The release process must not package or accept externally supplied
`master_list.pkl` files. No safe upstream cache replacement is included in the
fixed source baseline, so replacing the cache format remains an upgrade item;
release owners must record risk acceptance if this optional tool is enabled.
