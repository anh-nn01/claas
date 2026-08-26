# full traits
python create_dataset_task1.py --include_demo --demo_full --include_big5 --include_facet --include_locus
# no demographic traits
python create_dataset_task1.py --include_big5 --include_facet --include_locus
# no big five traits
python create_dataset_task1.py --include_demo --demo_full --include_facet --include_locus
# no facet traits
python create_dataset_task1.py --include_demo --demo_full --include_big5 --include_locus
# no locus of control
python create_dataset_task1.py --include_demo --demo_full --include_big5 --include_facet
# no big five & facet & locus traits
python create_dataset_task1.py --include_demo --demo_full