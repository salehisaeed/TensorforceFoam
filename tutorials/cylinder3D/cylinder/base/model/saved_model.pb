це
†х
D
AddV2
x"T
y"T
z"T"
Ttype:
2	АР
h
All	
input

reduction_indices"Tidx

output
"
	keep_dimsbool( "
Tidxtype0:
2	
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( И
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
T
CheckNumerics
tensor"T
output"T"
Ttype:
2"
messagestringИ
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
R
Equal
x"T
y"T
z
"	
Ttype"$
incompatible_shape_errorbool(Р
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
.
Identity

input"T
output"T"	
Ttype
Ї
If
cond"Tcond
input2Tin
output2Tout"
Tcondtype"
Tin
list(type)("
Tout
list(type)("
then_branchfunc"
else_branchfunc" 
output_shapeslist(shape)
 И
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
М
Max

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
>
Maximum
x"T
y"T
z"T"
Ttype:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(И

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
≥
PartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
f
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx" 
Tidxtype0:
2
	
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
Ѕ
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring И®
@
StaticRegexFullMatch	
input

output
"
patternstring
ч
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
-
Tanh
x"T
y"T"
Ttype:

2
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И
&
	ZerosLike
x"T
y"T"	
Ttype"serve*2.9.12v2.9.0-18-gd8ce9f9c3018ђ±
l
save_counterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_namesave_counter
e
 save_counter/Read/ReadVariableOpReadVariableOpsave_counter*
_output_shapes
: *
dtype0	
Ґ
#agent/policy/network/dense1/weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*4
shared_name%#agent/policy/network/dense1/weights
Ы
7agent/policy/network/dense1/weights/Read/ReadVariableOpReadVariableOp#agent/policy/network/dense1/weights*
_output_shapes

:@@*
dtype0
Ш
 agent/policy/network/dense1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*1
shared_name" agent/policy/network/dense1/bias
С
4agent/policy/network/dense1/bias/Read/ReadVariableOpReadVariableOp agent/policy/network/dense1/bias*
_output_shapes
:@*
dtype0
£
#agent/policy/network/dense0/weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ї@*4
shared_name%#agent/policy/network/dense0/weights
Ь
7agent/policy/network/dense0/weights/Read/ReadVariableOpReadVariableOp#agent/policy/network/dense0/weights*
_output_shapes
:	ї@*
dtype0
Ш
 agent/policy/network/dense0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*1
shared_name" agent/policy/network/dense0/bias
С
4agent/policy/network/dense0/bias/Read/ReadVariableOpReadVariableOp agent/policy/network/dense0/bias*
_output_shapes
:@*
dtype0
»
6agent/policy/action_distribution/stddev/linear/weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*G
shared_name86agent/policy/action_distribution/stddev/linear/weights
Ѕ
Jagent/policy/action_distribution/stddev/linear/weights/Read/ReadVariableOpReadVariableOp6agent/policy/action_distribution/stddev/linear/weights*
_output_shapes

:@*
dtype0
Њ
3agent/policy/action_distribution/stddev/linear/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*D
shared_name53agent/policy/action_distribution/stddev/linear/bias
Ј
Gagent/policy/action_distribution/stddev/linear/bias/Read/ReadVariableOpReadVariableOp3agent/policy/action_distribution/stddev/linear/bias*
_output_shapes
:*
dtype0
ƒ
4agent/policy/action_distribution/mean/linear/weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*E
shared_name64agent/policy/action_distribution/mean/linear/weights
љ
Hagent/policy/action_distribution/mean/linear/weights/Read/ReadVariableOpReadVariableOp4agent/policy/action_distribution/mean/linear/weights*
_output_shapes

:@*
dtype0
Ї
1agent/policy/action_distribution/mean/linear/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*B
shared_name31agent/policy/action_distribution/mean/linear/bias
≥
Eagent/policy/action_distribution/mean/linear/bias/Read/ReadVariableOpReadVariableOp1agent/policy/action_distribution/mean/linear/bias*
_output_shapes
:*
dtype0
ь
Tagent/policy/action_distribution/distributions_action_distribution-mean-summary-stepVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *e
shared_nameVTagent/policy/action_distribution/distributions_action_distribution-mean-summary-step
х
hagent/policy/action_distribution/distributions_action_distribution-mean-summary-step/Read/ReadVariableOpReadVariableOpTagent/policy/action_distribution/distributions_action_distribution-mean-summary-step*
_output_shapes
: *
dtype0	

NoOpNoOp
ђ
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*з
valueЁBЏ B”
ґ
ZVagent/policy/action_distribution/distributions_action_distribution-mean-summary-step:0
73agent/policy/action_distribution/mean/linear/bias:0
:6agent/policy/action_distribution/mean/linear/weights:0
95agent/policy/action_distribution/stddev/linear/bias:0
<8agent/policy/action_distribution/stddev/linear/weights:0
&"agent/policy/network/dense0/bias:0
)%agent/policy/network/dense0/weights:0
&"agent/policy/network/dense1/bias:0
)	%agent/policy/network/dense1/weights:0

save_counter
act
initial_internals

signatures*
г№
VARIABLE_VALUETagent/policy/action_distribution/distributions_action_distribution-mean-summary-steptagent.Spolicy.Saction_distribution.Sdistributions_action_distribution-mean-summary-step:0/.ATTRIBUTES/VARIABLE_VALUE*
ЯШ
VARIABLE_VALUE1agent/policy/action_distribution/mean/linear/biasSagent.Spolicy.Saction_distribution.Smean.Slinear.Sbias:0/.ATTRIBUTES/VARIABLE_VALUE*
•Ю
VARIABLE_VALUE4agent/policy/action_distribution/mean/linear/weightsVagent.Spolicy.Saction_distribution.Smean.Slinear.Sweights:0/.ATTRIBUTES/VARIABLE_VALUE*
£Ь
VARIABLE_VALUE3agent/policy/action_distribution/stddev/linear/biasUagent.Spolicy.Saction_distribution.Sstddev.Slinear.Sbias:0/.ATTRIBUTES/VARIABLE_VALUE*
©Ґ
VARIABLE_VALUE6agent/policy/action_distribution/stddev/linear/weightsXagent.Spolicy.Saction_distribution.Sstddev.Slinear.Sweights:0/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUE agent/policy/network/dense0/biasAagent.Spolicy.Snetwork.Sdense0.Sbias:0/.ATTRIBUTES/VARIABLE_VALUE*
Б{
VARIABLE_VALUE#agent/policy/network/dense0/weightsDagent.Spolicy.Snetwork.Sdense0.Sweights:0/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUE agent/policy/network/dense1/biasAagent.Spolicy.Snetwork.Sdense1.Sbias:0/.ATTRIBUTES/VARIABLE_VALUE*
Б{
VARIABLE_VALUE#agent/policy/network/dense1/weightsDagent.Spolicy.Snetwork.Sdense1.Sweights:0/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEsave_counter'save_counter/.ATTRIBUTES/VARIABLE_VALUE*

trace_0* 

trace_0* 

serving_default* 
* 
* 
* 
{
serving_default_args_0Placeholder*(
_output_shapes
:€€€€€€€€€ї*
dtype0*
shape:€€€€€€€€€ї
^
serving_default_deterministicPlaceholder*
_output_shapes
: *
dtype0
*
shape: 
њ
StatefulPartitionedCallStatefulPartitionedCallserving_default_args_0serving_default_deterministic#agent/policy/network/dense0/weights agent/policy/network/dense0/bias#agent/policy/network/dense1/weights agent/policy/network/dense1/bias4agent/policy/action_distribution/mean/linear/weights1agent/policy/action_distribution/mean/linear/bias6agent/policy/action_distribution/stddev/linear/weights3agent/policy/action_distribution/stddev/linear/bias*
Tin
2

*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:€€€€€€€€€**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8В *+
f&R$
"__inference_signature_wrapper_1429
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
≥
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamehagent/policy/action_distribution/distributions_action_distribution-mean-summary-step/Read/ReadVariableOpEagent/policy/action_distribution/mean/linear/bias/Read/ReadVariableOpHagent/policy/action_distribution/mean/linear/weights/Read/ReadVariableOpGagent/policy/action_distribution/stddev/linear/bias/Read/ReadVariableOpJagent/policy/action_distribution/stddev/linear/weights/Read/ReadVariableOp4agent/policy/network/dense0/bias/Read/ReadVariableOp7agent/policy/network/dense0/weights/Read/ReadVariableOp4agent/policy/network/dense1/bias/Read/ReadVariableOp7agent/policy/network/dense1/weights/Read/ReadVariableOp save_counter/Read/ReadVariableOpConst*
Tin
2		*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *&
f!R
__inference__traced_save_1892
ж
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameTagent/policy/action_distribution/distributions_action_distribution-mean-summary-step1agent/policy/action_distribution/mean/linear/bias4agent/policy/action_distribution/mean/linear/weights3agent/policy/action_distribution/stddev/linear/bias6agent/policy/action_distribution/stddev/linear/weights agent/policy/network/dense0/bias#agent/policy/network/dense0/weights agent/policy/network/dense1/bias#agent/policy/network/dense1/weightssave_counter*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *)
f$R"
 __inference__traced_restore_1932Яп
Ц
Ѓ
1agent_assert_equal_1_Assert_AssertGuard_true_1457M
Iagent_assert_equal_1_assert_assertguard_identity_agent_assert_equal_1_all
7
3agent_assert_equal_1_assert_assertguard_placeholder	9
5agent_assert_equal_1_assert_assertguard_placeholder_1	6
2agent_assert_equal_1_assert_assertguard_identity_1
J
,agent/assert_equal_1/Assert/AssertGuard/NoOpNoOp*
_output_shapes
 „
0agent/assert_equal_1/Assert/AssertGuard/IdentityIdentityIagent_assert_equal_1_assert_assertguard_identity_agent_assert_equal_1_all-^agent/assert_equal_1/Assert/AssertGuard/NoOp*
T0
*
_output_shapes
: Ъ
2agent/assert_equal_1/Assert/AssertGuard/Identity_1Identity9agent/assert_equal_1/Assert/AssertGuard/Identity:output:0*
T0
*
_output_shapes
: "q
2agent_assert_equal_1_assert_assertguard_identity_1;agent/assert_equal_1/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: ::: 

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:
Ф
ѕ
#action_distribution_cond_false_1322'
#action_distribution_cond_shape_mean'
#action_distribution_cond_mul_stddev,
(action_distribution_cond_mul_temperature%
!action_distribution_cond_identityИq
action_distribution/cond/ShapeShape#action_distribution_cond_shape_mean*
T0*
_output_shapes
:p
+action_distribution/cond/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    r
-action_distribution/cond/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Ј
;action_distribution/cond/random_normal/RandomStandardNormalRandomStandardNormal'action_distribution/cond/Shape:output:0*
T0*#
_output_shapes
:€€€€€€€€€*
dtype0Ё
*action_distribution/cond/random_normal/mulMulDaction_distribution/cond/random_normal/RandomStandardNormal:output:06action_distribution/cond/random_normal/stddev:output:0*
T0*#
_output_shapes
:€€€€€€€€€√
&action_distribution/cond/random_normalAddV2.action_distribution/cond/random_normal/mul:z:04action_distribution/cond/random_normal/mean:output:0*
T0*#
_output_shapes
:€€€€€€€€€†
action_distribution/cond/mulMul#action_distribution_cond_mul_stddev(action_distribution_cond_mul_temperature*
T0*#
_output_shapes
:€€€€€€€€€°
action_distribution/cond/mul_1Mul action_distribution/cond/mul:z:0*action_distribution/cond/random_normal:z:0*
T0*#
_output_shapes
:€€€€€€€€€Ь
action_distribution/cond/addAddV2#action_distribution_cond_shape_mean"action_distribution/cond/mul_1:z:0*
T0*#
_output_shapes
:€€€€€€€€€}
!action_distribution/cond/IdentityIdentity action_distribution/cond/add:z:0*
T0*#
_output_shapes
:€€€€€€€€€"O
!action_distribution_cond_identity*action_distribution/cond/Identity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :€€€€€€€€€:€€€€€€€€€: :) %
#
_output_shapes
:€€€€€€€€€:)%
#
_output_shapes
:€€€€€€€€€:

_output_shapes
: 
Щ!
Ж
__inference_parametrize_1695
x
	mean_1646:@
	mean_1648:
stddev_1672:@
stddev_1674:
identity

identity_1

identity_2ИҐmean/StatefulPartitionedCallҐstddev/StatefulPartitionedCall^
action_distribution/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж5`
action_distribution/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *U]Ѕ≠
mean/StatefulPartitionedCallStatefulPartitionedCallx	mean_1646	mean_1648*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *
fR
__inference_apply_1645t
!action_distribution/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€І
action_distribution/ReshapeReshape%mean/StatefulPartitionedCall:output:0*action_distribution/Reshape/shape:output:0*
T0*#
_output_shapes
:€€€€€€€€€≥
stddev/StatefulPartitionedCallStatefulPartitionedCallxstddev_1672stddev_1674*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *
fR
__inference_apply_1671v
#action_distribution/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€≠
action_distribution/Reshape_1Reshape'stddev/StatefulPartitionedCall:output:0,action_distribution/Reshape_1/shape:output:0*
T0*#
_output_shapes
:€€€€€€€€€e
action_distribution/NegNeg$action_distribution/Const_1:output:0*
T0*
_output_shapes
: І
)action_distribution/clip_by_value/MinimumMinimum&action_distribution/Reshape_1:output:0action_distribution/Neg:y:0*
T0*#
_output_shapes
:€€€€€€€€€ѓ
!action_distribution/clip_by_valueMaximum-action_distribution/clip_by_value/Minimum:z:0$action_distribution/Const_1:output:0*
T0*#
_output_shapes
:€€€€€€€€€`
action_distribution/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>`
action_distribution/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *r1?}
action_distribution/SoftplusSoftplus%action_distribution/clip_by_value:z:0*
T0*#
_output_shapes
:€€€€€€€€€†
action_distribution/addAddV2*action_distribution/Softplus:activations:0$action_distribution/Const_2:output:0*
T0*#
_output_shapes
:€€€€€€€€€П
action_distribution/add_1AddV2$action_distribution/Const_3:output:0$action_distribution/Const_2:output:0*
T0*
_output_shapes
: Р
action_distribution/truedivRealDivaction_distribution/add:z:0action_distribution/add_1:z:0*
T0*#
_output_shapes
:€€€€€€€€€`
action_distribution/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *  А>У
action_distribution/mulMulaction_distribution/truediv:z:0$action_distribution/Const_4:output:0*
T0*#
_output_shapes
:€€€€€€€€€С
action_distribution/add_2AddV2action_distribution/mul:z:0"action_distribution/Const:output:0*
T0*#
_output_shapes
:€€€€€€€€€k
action_distribution/LogLogaction_distribution/add_2:z:0*
T0*#
_output_shapes
:€€€€€€€€€o
IdentityIdentity$action_distribution/Reshape:output:0^NoOp*
T0*#
_output_shapes
:€€€€€€€€€h

Identity_1Identityaction_distribution/mul:z:0^NoOp*
T0*#
_output_shapes
:€€€€€€€€€h

Identity_2Identityaction_distribution/Log:y:0^NoOp*
T0*#
_output_shapes
:€€€€€€€€€Ж
NoOpNoOp^mean/StatefulPartitionedCall^stddev/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€@: : : : 2<
mean/StatefulPartitionedCallmean/StatefulPartitionedCall2@
stddev/StatefulPartitionedCallstddev/StatefulPartitionedCall:J F
'
_output_shapes
:€€€€€€€€€@

_user_specified_namex
©
Н
__inference_apply_1610

args_0
horizons	
deterministic

dense0_1583:	ї@
dense0_1585:@
dense1_1604:@@
dense1_1606:@
identityИҐdense0/StatefulPartitionedCallҐdense1/StatefulPartitionedCallЄ
dense0/StatefulPartitionedCallStatefulPartitionedCallargs_0dense0_1583dense0_1585*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *
fR
__inference_apply_1582ў
dense1/StatefulPartitionedCallStatefulPartitionedCall'dense0/StatefulPartitionedCall:output:0dense1_1604dense1_1606*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *
fR
__inference_apply_1603v
IdentityIdentity'dense1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@И
NoOpNoOp^dense0/StatefulPartitionedCall^dense1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:€€€€€€€€€ї:€€€€€€€€€: : : : : 2@
dense0/StatefulPartitionedCalldense0/StatefulPartitionedCall2@
dense1/StatefulPartitionedCalldense1/StatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€ї
 
_user_specified_nameargs_0:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
horizons:EA

_output_shapes
: 
'
_user_specified_namedeterministic
ћ
с
2agent_assert_equal_1_Assert_AssertGuard_false_1082K
Gagent_assert_equal_1_assert_assertguard_assert_agent_assert_equal_1_all
@
<agent_assert_equal_1_assert_assertguard_assert_agent_maximum	@
<agent_assert_equal_1_assert_assertguard_assert_agent_const_1	6
2agent_assert_equal_1_assert_assertguard_identity_1
ИҐ.agent/assert_equal_1/Assert/AssertGuard/Assert∞
5agent/assert_equal_1/Assert/AssertGuard/Assert/data_0Const*
_output_shapes
: *
dtype0*K
valueBB@ B:Policy/baseline on-policy horizon currently not supported.°
5agent/assert_equal_1/Assert/AssertGuard/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x == y did not hold element-wise:М
5agent/assert_equal_1/Assert/AssertGuard/Assert/data_2Const*
_output_shapes
: *
dtype0*'
valueB Bx (agent/Maximum:0) = М
5agent/assert_equal_1/Assert/AssertGuard/Assert/data_4Const*
_output_shapes
: *
dtype0*'
valueB By (agent/Const_1:0) = §
.agent/assert_equal_1/Assert/AssertGuard/AssertAssertGagent_assert_equal_1_assert_assertguard_assert_agent_assert_equal_1_all>agent/assert_equal_1/Assert/AssertGuard/Assert/data_0:output:0>agent/assert_equal_1/Assert/AssertGuard/Assert/data_1:output:0>agent/assert_equal_1/Assert/AssertGuard/Assert/data_2:output:0<agent_assert_equal_1_assert_assertguard_assert_agent_maximum>agent/assert_equal_1/Assert/AssertGuard/Assert/data_4:output:0<agent_assert_equal_1_assert_assertguard_assert_agent_const_1*
T

2		*
_output_shapes
 „
0agent/assert_equal_1/Assert/AssertGuard/IdentityIdentityGagent_assert_equal_1_assert_assertguard_assert_agent_assert_equal_1_all/^agent/assert_equal_1/Assert/AssertGuard/Assert*
T0
*
_output_shapes
: …
2agent/assert_equal_1/Assert/AssertGuard/Identity_1Identity9agent/assert_equal_1/Assert/AssertGuard/Identity:output:0-^agent/assert_equal_1/Assert/AssertGuard/NoOp*
T0
*
_output_shapes
: Я
,agent/assert_equal_1/Assert/AssertGuard/NoOpNoOp/^agent/assert_equal_1/Assert/AssertGuard/Assert*"
_acd_function_control_output(*
_output_shapes
 "q
2agent_assert_equal_1_assert_assertguard_identity_1;agent/assert_equal_1/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : 2`
.agent/assert_equal_1/Assert/AssertGuard/Assert.agent/assert_equal_1/Assert/AssertGuard/Assert: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Ц
Ѓ
1agent_assert_equal_1_Assert_AssertGuard_true_1008M
Iagent_assert_equal_1_assert_assertguard_identity_agent_assert_equal_1_all
7
3agent_assert_equal_1_assert_assertguard_placeholder	9
5agent_assert_equal_1_assert_assertguard_placeholder_1	6
2agent_assert_equal_1_assert_assertguard_identity_1
J
,agent/assert_equal_1/Assert/AssertGuard/NoOpNoOp*
_output_shapes
 „
0agent/assert_equal_1/Assert/AssertGuard/IdentityIdentityIagent_assert_equal_1_assert_assertguard_identity_agent_assert_equal_1_all-^agent/assert_equal_1/Assert/AssertGuard/NoOp*
T0
*
_output_shapes
: Ъ
2agent/assert_equal_1/Assert/AssertGuard/Identity_1Identity9agent/assert_equal_1/Assert/AssertGuard/Identity:output:0*
T0
*
_output_shapes
: "q
2agent_assert_equal_1_assert_assertguard_identity_1;agent/assert_equal_1/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: ::: 

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:
о
С
__inference_act_1795

args_0
horizons	
deterministic

network_1611:	ї@
network_1613:@
network_1615:@@
network_1617:@%
policy_cond_input_0:@!
policy_cond_input_1:%
policy_cond_input_2:@!
policy_cond_input_3:
identityИҐnetwork/StatefulPartitionedCallҐpolicy/condц
network/StatefulPartitionedCallStatefulPartitionedCallargs_0horizonsdeterministicnetwork_1611network_1613network_1615network_1617*
Tin
	2	
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *
fR
__inference_apply_1610≥
policy/condIfdeterministicpolicy_cond_input_0policy_cond_input_1policy_cond_input_2policy_cond_input_3(network/StatefulPartitionedCall:output:0*
Tcond0
*
Tin	
2*
Tout
2*
_lower_using_switch_merge(*#
_output_shapes
:€€€€€€€€€*&
_read_only_resource_inputs
*)
else_branchR
policy_cond_false_1621*"
output_shapes
:€€€€€€€€€*(
then_branchR
policy_cond_true_1620d
policy/cond/IdentityIdentitypolicy/cond:output:0*
T0*#
_output_shapes
:€€€€€€€€€h
IdentityIdentitypolicy/cond/Identity:output:0^NoOp*
T0*#
_output_shapes
:€€€€€€€€€v
NoOpNoOp ^network/StatefulPartitionedCall^policy/cond*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:€€€€€€€€€ї:€€€€€€€€€: : : : : : : : : 2B
network/StatefulPartitionedCallnetwork/StatefulPartitionedCall2
policy/condpolicy/cond:P L
(
_output_shapes
:€€€€€€€€€ї
 
_user_specified_nameargs_0:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
horizons:EA

_output_shapes
: 
'
_user_specified_namedeterministic
 
я
__inference_apply_1582
x8
%dense0_matmul_readvariableop_resource:	ї@4
&dense0_biasadd_readvariableop_resource:@
identityИҐdense0/BiasAdd/ReadVariableOpҐdense0/MatMul/ReadVariableOpГ
dense0/MatMul/ReadVariableOpReadVariableOp%dense0_matmul_readvariableop_resource*
_output_shapes
:	ї@*
dtype0r
dense0/MatMulMatMulx$dense0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@А
dense0/BiasAdd/ReadVariableOpReadVariableOp&dense0_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Л
dense0/BiasAddBiasAdddense0/MatMul:product:0%dense0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@Э
activation/PartitionedCallPartitionedCalldense0/BiasAdd:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *
fR
__inference_apply_1579r
IdentityIdentity#activation/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@Е
NoOpNoOp^dense0/BiasAdd/ReadVariableOp^dense0/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€ї: : 2>
dense0/BiasAdd/ReadVariableOpdense0/BiasAdd/ReadVariableOp2<
dense0/MatMul/ReadVariableOpdense0/MatMul/ReadVariableOp:K G
(
_output_shapes
:€€€€€€€€€ї

_user_specified_namex
№
-
__inference_past_horizon_1051
identity	O
network/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R [
network/stackPacknetwork/Const:output:0*
N*
T0	*
_output_shapes
:_
network/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : s
network/MaxMaxnetwork/stack:output:0&network/Max/reduction_indices:output:0*
T0	*
_output_shapes
: K
IdentityIdentitynetwork/Max:output:0*
T0	*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
Й
ф
policy_cond_false_1183*
action_distribution_1301:@&
action_distribution_1303:*
action_distribution_1305:@&
action_distribution_1307:7
3action_distribution_network_statefulpartitionedcall
policy_cond_identityИҐ+action_distribution/StatefulPartitionedCallҐ-action_distribution/StatefulPartitionedCall_1с
temperature/PartitionedCallPartitionedCall*	
Tin
 *
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *
fR
__inference_value_1299ж
+action_distribution/StatefulPartitionedCallStatefulPartitionedCall3action_distribution_network_statefulpartitionedcallaction_distribution_1301action_distribution_1303action_distribution_1305action_distribution_1307*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *%
f R
__inference_parametrize_1262з
-action_distribution/StatefulPartitionedCall_1StatefulPartitionedCall4action_distribution/StatefulPartitionedCall:output:04action_distribution/StatefulPartitionedCall:output:14action_distribution/StatefulPartitionedCall:output:2$temperature/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В * 
fR
__inference_sample_1355Щ
policy/cond/IdentityIdentity6action_distribution/StatefulPartitionedCall_1:output:0^policy/cond/NoOp*
T0*#
_output_shapes
:€€€€€€€€€∞
policy/cond/NoOpNoOp,^action_distribution/StatefulPartitionedCall.^action_distribution/StatefulPartitionedCall_1*"
_acd_function_control_output(*
_output_shapes
 "5
policy_cond_identitypolicy/cond/Identity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
: : : : :€€€€€€€€€@2Z
+action_distribution/StatefulPartitionedCall+action_distribution/StatefulPartitionedCall2^
-action_distribution/StatefulPartitionedCall_1-action_distribution/StatefulPartitionedCall_1:-)
'
_output_shapes
:€€€€€€€€€@
№
-
__inference_past_horizon_1509
identity	O
network/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R [
network/stackPacknetwork/Const:output:0*
N*
T0	*
_output_shapes
:_
network/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : s
network/MaxMaxnetwork/stack:output:0&network/Max/reduction_indices:output:0*
T0	*
_output_shapes
: K
IdentityIdentitynetwork/Max:output:0*
T0	*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
а
|
__inference_sample_1785
mean

stddev

log_stddev
temperature
identityИҐaction_distribution/cond^
action_distribution/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж5r
action_distribution/LessLesstemperature"action_distribution/Const:output:0*
T0*
_output_shapes
: ю
action_distribution/condIfaction_distribution/Less:z:0meanstddevtemperature*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*#
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *6
else_branch'R%
#action_distribution_cond_false_1752*"
output_shapes
:€€€€€€€€€*5
then_branch&R$
"action_distribution_cond_true_1751~
!action_distribution/cond/IdentityIdentity!action_distribution/cond:output:0*
T0*#
_output_shapes
:€€€€€€€€€`
action_distribution/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *  А?z
action_distribution/TanhTanh*action_distribution/cond/Identity:output:0*
T0*#
_output_shapes
:€€€€€€€€€`
action_distribution/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *   ?`
action_distribution/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *  Ањ`
action_distribution/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *  А?Л
action_distribution/subSub$action_distribution/Const_4:output:0$action_distribution/Const_3:output:0*
T0*
_output_shapes
: В
action_distribution/mulMulaction_distribution/sub:z:0$action_distribution/Const_2:output:0*
T0*
_output_shapes
: Т
action_distribution/addAddV2action_distribution/Tanh:y:0$action_distribution/Const_1:output:0*
T0*#
_output_shapes
:€€€€€€€€€И
action_distribution/mul_1Mulaction_distribution/mul:z:0action_distribution/add:z:0*
T0*#
_output_shapes
:€€€€€€€€€Х
action_distribution/add_1AddV2$action_distribution/Const_3:output:0action_distribution/mul_1:z:0*
T0*#
_output_shapes
:€€€€€€€€€h
IdentityIdentityaction_distribution/add_1:z:0^NoOp*
T0*#
_output_shapes
:€€€€€€€€€a
NoOpNoOp^action_distribution/cond*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€: 24
action_distribution/condaction_distribution/cond:I E
#
_output_shapes
:€€€€€€€€€

_user_specified_namemean:KG
#
_output_shapes
:€€€€€€€€€
 
_user_specified_namestddev:OK
#
_output_shapes
:€€€€€€€€€
$
_user_specified_name
log_stddev:C?

_output_shapes
: 
%
_user_specified_nametemperature
ъ
Ќ
"action_distribution_cond_true_1321*
&action_distribution_cond_identity_mean(
$action_distribution_cond_placeholder*
&action_distribution_cond_placeholder_1%
!action_distribution_cond_identityГ
!action_distribution/cond/IdentityIdentity&action_distribution_cond_identity_mean*
T0*#
_output_shapes
:€€€€€€€€€"O
!action_distribution_cond_identity*action_distribution/cond/Identity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :€€€€€€€€€:€€€€€€€€€: :) %
#
_output_shapes
:€€€€€€€€€:)%
#
_output_shapes
:€€€€€€€€€:

_output_shapes
: 
Щ!
Ж
__inference_parametrize_1262
x
	mean_1211:@
	mean_1213:
stddev_1239:@
stddev_1241:
identity

identity_1

identity_2ИҐmean/StatefulPartitionedCallҐstddev/StatefulPartitionedCall^
action_distribution/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж5`
action_distribution/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *U]Ѕ≠
mean/StatefulPartitionedCallStatefulPartitionedCallx	mean_1211	mean_1213*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *
fR
__inference_apply_1210t
!action_distribution/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€І
action_distribution/ReshapeReshape%mean/StatefulPartitionedCall:output:0*action_distribution/Reshape/shape:output:0*
T0*#
_output_shapes
:€€€€€€€€€≥
stddev/StatefulPartitionedCallStatefulPartitionedCallxstddev_1239stddev_1241*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *
fR
__inference_apply_1238v
#action_distribution/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€≠
action_distribution/Reshape_1Reshape'stddev/StatefulPartitionedCall:output:0,action_distribution/Reshape_1/shape:output:0*
T0*#
_output_shapes
:€€€€€€€€€e
action_distribution/NegNeg$action_distribution/Const_1:output:0*
T0*
_output_shapes
: І
)action_distribution/clip_by_value/MinimumMinimum&action_distribution/Reshape_1:output:0action_distribution/Neg:y:0*
T0*#
_output_shapes
:€€€€€€€€€ѓ
!action_distribution/clip_by_valueMaximum-action_distribution/clip_by_value/Minimum:z:0$action_distribution/Const_1:output:0*
T0*#
_output_shapes
:€€€€€€€€€`
action_distribution/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>`
action_distribution/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *r1?}
action_distribution/SoftplusSoftplus%action_distribution/clip_by_value:z:0*
T0*#
_output_shapes
:€€€€€€€€€†
action_distribution/addAddV2*action_distribution/Softplus:activations:0$action_distribution/Const_2:output:0*
T0*#
_output_shapes
:€€€€€€€€€П
action_distribution/add_1AddV2$action_distribution/Const_3:output:0$action_distribution/Const_2:output:0*
T0*
_output_shapes
: Р
action_distribution/truedivRealDivaction_distribution/add:z:0action_distribution/add_1:z:0*
T0*#
_output_shapes
:€€€€€€€€€`
action_distribution/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *  А>У
action_distribution/mulMulaction_distribution/truediv:z:0$action_distribution/Const_4:output:0*
T0*#
_output_shapes
:€€€€€€€€€С
action_distribution/add_2AddV2action_distribution/mul:z:0"action_distribution/Const:output:0*
T0*#
_output_shapes
:€€€€€€€€€k
action_distribution/LogLogaction_distribution/add_2:z:0*
T0*#
_output_shapes
:€€€€€€€€€o
IdentityIdentity$action_distribution/Reshape:output:0^NoOp*
T0*#
_output_shapes
:€€€€€€€€€h

Identity_1Identityaction_distribution/mul:z:0^NoOp*
T0*#
_output_shapes
:€€€€€€€€€h

Identity_2Identityaction_distribution/Log:y:0^NoOp*
T0*#
_output_shapes
:€€€€€€€€€Ж
NoOpNoOp^mean/StatefulPartitionedCall^stddev/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€@: : : : 2<
mean/StatefulPartitionedCallmean/StatefulPartitionedCall2@
stddev/StatefulPartitionedCallstddev/StatefulPartitionedCall:J F
'
_output_shapes
:€€€€€€€€€@

_user_specified_namex
®
Л
__inference_apply_1238
x
linear_1232:@
linear_1234:
identityИҐlinear/StatefulPartitionedCall≥
linear/StatefulPartitionedCallStatefulPartitionedCallxlinear_1232linear_1234*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *
fR
__inference_apply_1231v
IdentityIdentity'linear/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€g
NoOpNoOp^linear/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€@: : 2@
linear/StatefulPartitionedCalllinear/StatefulPartitionedCall:J F
'
_output_shapes
:€€€€€€€€€@

_user_specified_namex
ъ	
≈
"__inference_signature_wrapper_1429

args_0
deterministic

unknown:	ї@
	unknown_0:@
	unknown_1:@@
	unknown_2:@
	unknown_3:@
	unknown_4:
	unknown_5:@
	unknown_6:
identityИҐStatefulPartitionedCallП
StatefulPartitionedCallStatefulPartitionedCallargs_0deterministicunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2

*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:€€€€€€€€€**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8В *)
f$R"
 __inference_independent_act_1405k
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*#
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€ї: : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€ї
 
_user_specified_nameargs_0:EA

_output_shapes
: 
'
_user_specified_namedeterministic
“
п
2agent_assert_equal_1_Assert_AssertGuard_false_1458K
Gagent_assert_equal_1_assert_assertguard_assert_agent_assert_equal_1_all
?
;agent_assert_equal_1_assert_assertguard_assert_agent_cast_1	?
;agent_assert_equal_1_assert_assertguard_assert_agent_concat	6
2agent_assert_equal_1_assert_assertguard_identity_1
ИҐ.agent/assert_equal_1/Assert/AssertGuard/Assertђ
5agent/assert_equal_1/Assert/AssertGuard/Assert/data_0Const*
_output_shapes
: *
dtype0*G
value>B< B6Agent.independent_act: invalid shape for  state input.°
5agent/assert_equal_1/Assert/AssertGuard/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x == y did not hold element-wise:Л
5agent/assert_equal_1/Assert/AssertGuard/Assert/data_2Const*
_output_shapes
: *
dtype0*&
valueB Bx (agent/Cast_1:0) = Л
5agent/assert_equal_1/Assert/AssertGuard/Assert/data_4Const*
_output_shapes
: *
dtype0*&
valueB By (agent/concat:0) = Ґ
.agent/assert_equal_1/Assert/AssertGuard/AssertAssertGagent_assert_equal_1_assert_assertguard_assert_agent_assert_equal_1_all>agent/assert_equal_1/Assert/AssertGuard/Assert/data_0:output:0>agent/assert_equal_1/Assert/AssertGuard/Assert/data_1:output:0>agent/assert_equal_1/Assert/AssertGuard/Assert/data_2:output:0;agent_assert_equal_1_assert_assertguard_assert_agent_cast_1>agent/assert_equal_1/Assert/AssertGuard/Assert/data_4:output:0;agent_assert_equal_1_assert_assertguard_assert_agent_concat*
T

2		*
_output_shapes
 „
0agent/assert_equal_1/Assert/AssertGuard/IdentityIdentityGagent_assert_equal_1_assert_assertguard_assert_agent_assert_equal_1_all/^agent/assert_equal_1/Assert/AssertGuard/Assert*
T0
*
_output_shapes
: …
2agent/assert_equal_1/Assert/AssertGuard/Identity_1Identity9agent/assert_equal_1/Assert/AssertGuard/Identity:output:0-^agent/assert_equal_1/Assert/AssertGuard/NoOp*
T0
*
_output_shapes
: Я
,agent/assert_equal_1/Assert/AssertGuard/NoOpNoOp/^agent/assert_equal_1/Assert/AssertGuard/Assert*"
_acd_function_control_output(*
_output_shapes
 "q
2agent_assert_equal_1_assert_assertguard_identity_1;agent/assert_equal_1/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: ::2`
.agent/assert_equal_1/Assert/AssertGuard/Assert.agent/assert_equal_1/Assert/AssertGuard/Assert: 

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:
Ч
-
__inference_apply_1139
x
identityL
activation/TanhTanhx*
T0*'
_output_shapes
:€€€€€€€€€@[
IdentityIdentityactivation/Tanh:y:0*
T0*'
_output_shapes
:€€€€€€€€€@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:€€€€€€€€€@:J F
'
_output_shapes
:€€€€€€€€€@

_user_specified_namex
Ж
Ѓ
1agent_assert_equal_1_Assert_AssertGuard_true_1525M
Iagent_assert_equal_1_assert_assertguard_identity_agent_assert_equal_1_all
7
3agent_assert_equal_1_assert_assertguard_placeholder	9
5agent_assert_equal_1_assert_assertguard_placeholder_1	6
2agent_assert_equal_1_assert_assertguard_identity_1
J
,agent/assert_equal_1/Assert/AssertGuard/NoOpNoOp*
_output_shapes
 „
0agent/assert_equal_1/Assert/AssertGuard/IdentityIdentityIagent_assert_equal_1_assert_assertguard_identity_agent_assert_equal_1_all-^agent/assert_equal_1/Assert/AssertGuard/NoOp*
T0
*
_output_shapes
: Ъ
2agent/assert_equal_1/Assert/AssertGuard/Identity_1Identity9agent/assert_equal_1/Assert/AssertGuard/Identity:output:0*
T0
*
_output_shapes
: "q
2agent_assert_equal_1_assert_assertguard_identity_1;agent/assert_equal_1/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : : 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
®
Л
__inference_apply_1645
x
linear_1639:@
linear_1641:
identityИҐlinear/StatefulPartitionedCall≥
linear/StatefulPartitionedCallStatefulPartitionedCallxlinear_1639linear_1641*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *
fR
__inference_apply_1638v
IdentityIdentity'linear/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€g
NoOpNoOp^linear/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€@: : 2@
linear/StatefulPartitionedCalllinear/StatefulPartitionedCall:J F
'
_output_shapes
:€€€€€€€€€@

_user_specified_namex
…
K
__inference_mode_1291
mean

stddev

log_stddev
identity^
action_distribution/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?T
action_distribution/TanhTanhmean*
T0*#
_output_shapes
:€€€€€€€€€`
action_distribution/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?`
action_distribution/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *  Ањ`
action_distribution/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *  А?Л
action_distribution/subSub$action_distribution/Const_3:output:0$action_distribution/Const_2:output:0*
T0*
_output_shapes
: В
action_distribution/mulMulaction_distribution/sub:z:0$action_distribution/Const_1:output:0*
T0*
_output_shapes
: Р
action_distribution/addAddV2action_distribution/Tanh:y:0"action_distribution/Const:output:0*
T0*#
_output_shapes
:€€€€€€€€€И
action_distribution/mul_1Mulaction_distribution/mul:z:0action_distribution/add:z:0*
T0*#
_output_shapes
:€€€€€€€€€Х
action_distribution/add_1AddV2$action_distribution/Const_2:output:0action_distribution/mul_1:z:0*
T0*#
_output_shapes
:€€€€€€€€€a
IdentityIdentityaction_distribution/add_1:z:0*
T0*#
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:I E
#
_output_shapes
:€€€€€€€€€

_user_specified_namemean:KG
#
_output_shapes
:€€€€€€€€€
 
_user_specified_namestddev:OK
#
_output_shapes
:€€€€€€€€€
$
_user_specified_name
log_stddev
—1
∞
 __inference__traced_restore_1932
file_prefixo
eassignvariableop_agent_policy_action_distribution_distributions_action_distribution_mean_summary_step:	 R
Dassignvariableop_1_agent_policy_action_distribution_mean_linear_bias:Y
Gassignvariableop_2_agent_policy_action_distribution_mean_linear_weights:@T
Fassignvariableop_3_agent_policy_action_distribution_stddev_linear_bias:[
Iassignvariableop_4_agent_policy_action_distribution_stddev_linear_weights:@A
3assignvariableop_5_agent_policy_network_dense0_bias:@I
6assignvariableop_6_agent_policy_network_dense0_weights:	ї@A
3assignvariableop_7_agent_policy_network_dense1_bias:@H
6assignvariableop_8_agent_policy_network_dense1_weights:@@)
assignvariableop_9_save_counter:	 
identity_11ИҐAssignVariableOpҐAssignVariableOp_1ҐAssignVariableOp_2ҐAssignVariableOp_3ҐAssignVariableOp_4ҐAssignVariableOp_5ҐAssignVariableOp_6ҐAssignVariableOp_7ҐAssignVariableOp_8ҐAssignVariableOp_9Ь
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*¬
valueЄBµBtagent.Spolicy.Saction_distribution.Sdistributions_action_distribution-mean-summary-step:0/.ATTRIBUTES/VARIABLE_VALUEBSagent.Spolicy.Saction_distribution.Smean.Slinear.Sbias:0/.ATTRIBUTES/VARIABLE_VALUEBVagent.Spolicy.Saction_distribution.Smean.Slinear.Sweights:0/.ATTRIBUTES/VARIABLE_VALUEBUagent.Spolicy.Saction_distribution.Sstddev.Slinear.Sbias:0/.ATTRIBUTES/VARIABLE_VALUEBXagent.Spolicy.Saction_distribution.Sstddev.Slinear.Sweights:0/.ATTRIBUTES/VARIABLE_VALUEBAagent.Spolicy.Snetwork.Sdense0.Sbias:0/.ATTRIBUTES/VARIABLE_VALUEBDagent.Spolicy.Snetwork.Sdense0.Sweights:0/.ATTRIBUTES/VARIABLE_VALUEBAagent.Spolicy.Snetwork.Sdense1.Sbias:0/.ATTRIBUTES/VARIABLE_VALUEBDagent.Spolicy.Snetwork.Sdense1.Sweights:0/.ATTRIBUTES/VARIABLE_VALUEB'save_counter/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHЖ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*)
value BB B B B B B B B B B B ’
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*@
_output_shapes.
,:::::::::::*
dtypes
2		[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0	*
_output_shapes
:–
AssignVariableOpAssignVariableOpeassignvariableop_agent_policy_action_distribution_distributions_action_distribution_mean_summary_stepIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:≥
AssignVariableOp_1AssignVariableOpDassignvariableop_1_agent_policy_action_distribution_mean_linear_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:ґ
AssignVariableOp_2AssignVariableOpGassignvariableop_2_agent_policy_action_distribution_mean_linear_weightsIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:µ
AssignVariableOp_3AssignVariableOpFassignvariableop_3_agent_policy_action_distribution_stddev_linear_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:Є
AssignVariableOp_4AssignVariableOpIassignvariableop_4_agent_policy_action_distribution_stddev_linear_weightsIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:Ґ
AssignVariableOp_5AssignVariableOp3assignvariableop_5_agent_policy_network_dense0_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:•
AssignVariableOp_6AssignVariableOp6assignvariableop_6_agent_policy_network_dense0_weightsIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:Ґ
AssignVariableOp_7AssignVariableOp3assignvariableop_7_agent_policy_network_dense1_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:•
AssignVariableOp_8AssignVariableOp6assignvariableop_8_agent_policy_network_dense1_weightsIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0	*
_output_shapes
:О
AssignVariableOp_9AssignVariableOpassignvariableop_9_save_counterIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	1
NoOpNoOp"/device:CPU:0*
_output_shapes
 Ђ
Identity_10Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_11IdentityIdentity_10:output:0^NoOp_1*
T0*
_output_shapes
: Ш
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_11Identity_11:output:0*)
_input_shapes
: : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
 
я
__inference_apply_1142
x8
%dense0_matmul_readvariableop_resource:	ї@4
&dense0_biasadd_readvariableop_resource:@
identityИҐdense0/BiasAdd/ReadVariableOpҐdense0/MatMul/ReadVariableOpГ
dense0/MatMul/ReadVariableOpReadVariableOp%dense0_matmul_readvariableop_resource*
_output_shapes
:	ї@*
dtype0r
dense0/MatMulMatMulx$dense0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@А
dense0/BiasAdd/ReadVariableOpReadVariableOp&dense0_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Л
dense0/BiasAddBiasAdddense0/MatMul:product:0%dense0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@Э
activation/PartitionedCallPartitionedCalldense0/BiasAdd:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *
fR
__inference_apply_1139r
IdentityIdentity#activation/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@Е
NoOpNoOp^dense0/BiasAdd/ReadVariableOp^dense0/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€ї: : 2>
dense0/BiasAdd/ReadVariableOpdense0/BiasAdd/ReadVariableOp2<
dense0/MatMul/ReadVariableOpdense0/MatMul/ReadVariableOp:K G
(
_output_shapes
:€€€€€€€€€ї

_user_specified_namex
о
С
__inference_act_1365

args_0
horizons	
deterministic

network_1173:	ї@
network_1175:@
network_1177:@@
network_1179:@%
policy_cond_input_0:@!
policy_cond_input_1:%
policy_cond_input_2:@!
policy_cond_input_3:
identityИҐnetwork/StatefulPartitionedCallҐpolicy/condц
network/StatefulPartitionedCallStatefulPartitionedCallargs_0horizonsdeterministicnetwork_1173network_1175network_1177network_1179*
Tin
	2	
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *
fR
__inference_apply_1172≥
policy/condIfdeterministicpolicy_cond_input_0policy_cond_input_1policy_cond_input_2policy_cond_input_3(network/StatefulPartitionedCall:output:0*
Tcond0
*
Tin	
2*
Tout
2*
_lower_using_switch_merge(*#
_output_shapes
:€€€€€€€€€*&
_read_only_resource_inputs
*)
else_branchR
policy_cond_false_1183*"
output_shapes
:€€€€€€€€€*(
then_branchR
policy_cond_true_1182d
policy/cond/IdentityIdentitypolicy/cond:output:0*
T0*#
_output_shapes
:€€€€€€€€€h
IdentityIdentitypolicy/cond/Identity:output:0^NoOp*
T0*#
_output_shapes
:€€€€€€€€€v
NoOpNoOp ^network/StatefulPartitionedCall^policy/cond*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:€€€€€€€€€ї:€€€€€€€€€: : : : : : : : : 2B
network/StatefulPartitionedCallnetwork/StatefulPartitionedCall2
policy/condpolicy/cond:P L
(
_output_shapes
:€€€€€€€€€ї
 
_user_specified_nameargs_0:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
horizons:EA

_output_shapes
: 
'
_user_specified_namedeterministic
…
K
__inference_mode_1723
mean

stddev

log_stddev
identity^
action_distribution/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?T
action_distribution/TanhTanhmean*
T0*#
_output_shapes
:€€€€€€€€€`
action_distribution/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?`
action_distribution/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *  Ањ`
action_distribution/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *  А?Л
action_distribution/subSub$action_distribution/Const_3:output:0$action_distribution/Const_2:output:0*
T0*
_output_shapes
: В
action_distribution/mulMulaction_distribution/sub:z:0$action_distribution/Const_1:output:0*
T0*
_output_shapes
: Р
action_distribution/addAddV2action_distribution/Tanh:y:0"action_distribution/Const:output:0*
T0*#
_output_shapes
:€€€€€€€€€И
action_distribution/mul_1Mulaction_distribution/mul:z:0action_distribution/add:z:0*
T0*#
_output_shapes
:€€€€€€€€€Х
action_distribution/add_1AddV2$action_distribution/Const_2:output:0action_distribution/mul_1:z:0*
T0*#
_output_shapes
:€€€€€€€€€a
IdentityIdentityaction_distribution/add_1:z:0*
T0*#
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:I E
#
_output_shapes
:€€€€€€€€€

_user_specified_namemean:KG
#
_output_shapes
:€€€€€€€€€
 
_user_specified_namestddev:OK
#
_output_shapes
:€€€€€€€€€
$
_user_specified_name
log_stddev
ґ.
†
 __inference_independent_act_1835

args_0
deterministic


agent_1817:	ї@

agent_1819:@

agent_1821:@@

agent_1823:@

agent_1825:@

agent_1827:

agent_1829:@

agent_1831:
identityИҐagent/StatefulPartitionedCallҐ agent/VerifyFinite/CheckNumericsҐ'agent/assert_equal_1/Assert/AssertGuardA
agent/ShapeShapeargs_0*
T0*
_output_shapes
:c
agent/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: e
agent/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:e
agent/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:п
agent/strided_sliceStridedSliceagent/Shape:output:0"agent/strided_slice/stack:output:0$agent/strided_slice/stack_1:output:0$agent/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask`

agent/CastCastagent/strided_slice:output:0*

DstT0	*

SrcT0*
_output_shapes
: M
agent/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 ZR
4agent/assert_type/statically_determined_correct_typeNoOp*
_output_shapes
 X
agent/Const_1Const*
_output_shapes
:*
dtype0	*
valueB	RїV
agent/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : r
agent/ExpandDims
ExpandDimsagent/Cast:y:0agent/ExpandDims/dim:output:0*
T0	*
_output_shapes
:S
agent/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Х
agent/concatConcatV2agent/ExpandDims:output:0agent/Const_1:output:0agent/concat/axis:output:0*
N*
T0	*
_output_shapes
:C
agent/Shape_1Shapeargs_0*
T0*
_output_shapes
:`
agent/Cast_1Castagent/Shape_1:output:0*

DstT0	*

SrcT0*
_output_shapes
:q
agent/assert_equal_1/EqualEqualagent/Cast_1:y:0agent/concat:output:0*
T0	*
_output_shapes
:d
agent/assert_equal_1/ConstConst*
_output_shapes
:*
dtype0*
valueB: |
agent/assert_equal_1/AllAllagent/assert_equal_1/Equal:z:0#agent/assert_equal_1/Const:output:0*
_output_shapes
: Ш
!agent/assert_equal_1/Assert/ConstConst*
_output_shapes
: *
dtype0*G
value>B< B6Agent.independent_act: invalid shape for  state input.П
#agent/assert_equal_1/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x == y did not hold element-wise:y
#agent/assert_equal_1/Assert/Const_2Const*
_output_shapes
: *
dtype0*&
valueB Bx (agent/Cast_1:0) = y
#agent/assert_equal_1/Assert/Const_3Const*
_output_shapes
: *
dtype0*&
valueB By (agent/concat:0) = «
'agent/assert_equal_1/Assert/AssertGuardIf!agent/assert_equal_1/All:output:0!agent/assert_equal_1/All:output:0agent/Cast_1:y:0agent/concat:output:0*
Tcond0
*
Tin
2
		*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *E
else_branch6R4
2agent_assert_equal_1_Assert_AssertGuard_false_1458*
output_shapes
: *D
then_branch5R3
1agent_assert_equal_1_Assert_AssertGuard_true_1457П
0agent/assert_equal_1/Assert/AssertGuard/IdentityIdentity0agent/assert_equal_1/Assert/AssertGuard:output:0*
T0
*
_output_shapes
: ю
 agent/VerifyFinite/CheckNumericsCheckNumericsargs_0(^agent/assert_equal_1/Assert/AssertGuard*
T0*
_class
loc:@args_0*(
_output_shapes
:€€€€€€€€€ї*K
message@>Agent.independent_act: invalid inf/nan value for  state input.™
%agent/VerifyFinite/control_dependencyIdentityargs_0!^agent/VerifyFinite/CheckNumerics*
T0*
_class
loc:@args_0*(
_output_shapes
:€€€€€€€€€їT
6agent/assert_type_1/statically_determined_correct_typeNoOp*
_output_shapes
 P
agent/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ≥
agent/ExpandDims_1/dimConst&^agent/VerifyFinite/control_dependency1^agent/assert_equal_1/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
value	B : v
agent/ExpandDims_1
ExpandDimsagent/Cast:y:0agent/ExpandDims_1/dim:output:0*
T0	*
_output_shapes
:Ѓ
agent/zeros/ConstConst&^agent/VerifyFinite/control_dependency1^agent/assert_equal_1/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R М
agent/zerosFillagent/ExpandDims_1:output:0agent/zeros/Const:output:0*
T0	*#
_output_shapes
:€€€€€€€€€*

index_type0	ѓ
agent/StatefulPartitionedCallStatefulPartitionedCallargs_0agent/zeros:output:0deterministic
agent_1817
agent_1819
agent_1821
agent_1823
agent_1825
agent_1827
agent_1829
agent_1831*
Tin
2	
*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:€€€€€€€€€**
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8В *"
fR
__inference_core_act_1816q
IdentityIdentity&agent/StatefulPartitionedCall:output:0^NoOp*
T0*#
_output_shapes
:€€€€€€€€€≥
NoOpNoOp^agent/StatefulPartitionedCall!^agent/VerifyFinite/CheckNumerics(^agent/assert_equal_1/Assert/AssertGuard*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€ї: : : : : : : : : 2>
agent/StatefulPartitionedCallagent/StatefulPartitionedCall2D
 agent/VerifyFinite/CheckNumerics agent/VerifyFinite/CheckNumerics2R
'agent/assert_equal_1/Assert/AssertGuard'agent/assert_equal_1/Assert/AssertGuard:P L
(
_output_shapes
:€€€€€€€€€ї
 
_user_specified_nameargs_0:EA

_output_shapes
: 
'
_user_specified_namedeterministic
џ
-
__inference_past_horizon_1054
identity	ф
network/PartitionedCallPartitionedCall*	
Tin
 *
Tout
2	*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *&
f!R
__inference_past_horizon_1051W
IdentityIdentity network/PartitionedCall:output:0*
T0	*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
ѓ
&
__inference_value_1299
identityV
temperature/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Q
IdentityIdentitytemperature/Const:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
а
|
__inference_sample_1355
mean

stddev

log_stddev
temperature
identityИҐaction_distribution/cond^
action_distribution/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж5r
action_distribution/LessLesstemperature"action_distribution/Const:output:0*
T0*
_output_shapes
: ю
action_distribution/condIfaction_distribution/Less:z:0meanstddevtemperature*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*#
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *6
else_branch'R%
#action_distribution_cond_false_1322*"
output_shapes
:€€€€€€€€€*5
then_branch&R$
"action_distribution_cond_true_1321~
!action_distribution/cond/IdentityIdentity!action_distribution/cond:output:0*
T0*#
_output_shapes
:€€€€€€€€€`
action_distribution/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *  А?z
action_distribution/TanhTanh*action_distribution/cond/Identity:output:0*
T0*#
_output_shapes
:€€€€€€€€€`
action_distribution/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *   ?`
action_distribution/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *  Ањ`
action_distribution/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *  А?Л
action_distribution/subSub$action_distribution/Const_4:output:0$action_distribution/Const_3:output:0*
T0*
_output_shapes
: В
action_distribution/mulMulaction_distribution/sub:z:0$action_distribution/Const_2:output:0*
T0*
_output_shapes
: Т
action_distribution/addAddV2action_distribution/Tanh:y:0$action_distribution/Const_1:output:0*
T0*#
_output_shapes
:€€€€€€€€€И
action_distribution/mul_1Mulaction_distribution/mul:z:0action_distribution/add:z:0*
T0*#
_output_shapes
:€€€€€€€€€Х
action_distribution/add_1AddV2$action_distribution/Const_3:output:0action_distribution/mul_1:z:0*
T0*#
_output_shapes
:€€€€€€€€€h
IdentityIdentityaction_distribution/add_1:z:0^NoOp*
T0*#
_output_shapes
:€€€€€€€€€a
NoOpNoOp^action_distribution/cond*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€: 24
action_distribution/condaction_distribution/cond:I E
#
_output_shapes
:€€€€€€€€€

_user_specified_namemean:KG
#
_output_shapes
:€€€€€€€€€
 
_user_specified_namestddev:OK
#
_output_shapes
:€€€€€€€€€
$
_user_specified_name
log_stddev:C?

_output_shapes
: 
%
_user_specified_nametemperature
Ф
ѕ
#action_distribution_cond_false_1752'
#action_distribution_cond_shape_mean'
#action_distribution_cond_mul_stddev,
(action_distribution_cond_mul_temperature%
!action_distribution_cond_identityИq
action_distribution/cond/ShapeShape#action_distribution_cond_shape_mean*
T0*
_output_shapes
:p
+action_distribution/cond/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    r
-action_distribution/cond/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Ј
;action_distribution/cond/random_normal/RandomStandardNormalRandomStandardNormal'action_distribution/cond/Shape:output:0*
T0*#
_output_shapes
:€€€€€€€€€*
dtype0Ё
*action_distribution/cond/random_normal/mulMulDaction_distribution/cond/random_normal/RandomStandardNormal:output:06action_distribution/cond/random_normal/stddev:output:0*
T0*#
_output_shapes
:€€€€€€€€€√
&action_distribution/cond/random_normalAddV2.action_distribution/cond/random_normal/mul:z:04action_distribution/cond/random_normal/mean:output:0*
T0*#
_output_shapes
:€€€€€€€€€†
action_distribution/cond/mulMul#action_distribution_cond_mul_stddev(action_distribution_cond_mul_temperature*
T0*#
_output_shapes
:€€€€€€€€€°
action_distribution/cond/mul_1Mul action_distribution/cond/mul:z:0*action_distribution/cond/random_normal:z:0*
T0*#
_output_shapes
:€€€€€€€€€Ь
action_distribution/cond/addAddV2#action_distribution_cond_shape_mean"action_distribution/cond/mul_1:z:0*
T0*#
_output_shapes
:€€€€€€€€€}
!action_distribution/cond/IdentityIdentity action_distribution/cond/add:z:0*
T0*#
_output_shapes
:€€€€€€€€€"O
!action_distribution_cond_identity*action_distribution/cond/Identity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :€€€€€€€€€:€€€€€€€€€: :) %
#
_output_shapes
:€€€€€€€€€:)%
#
_output_shapes
:€€€€€€€€€:

_output_shapes
: 
®
Л
__inference_apply_1210
x
linear_1204:@
linear_1206:
identityИҐlinear/StatefulPartitionedCall≥
linear/StatefulPartitionedCallStatefulPartitionedCallxlinear_1204linear_1206*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *
fR
__inference_apply_1203v
IdentityIdentity'linear/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€g
NoOpNoOp^linear/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€@: : 2@
linear/StatefulPartitionedCalllinear/StatefulPartitionedCall:J F
'
_output_shapes
:€€€€€€€€€@

_user_specified_namex
П
√
policy_cond_true_1620*
action_distribution_1696:@&
action_distribution_1698:*
action_distribution_1700:@&
action_distribution_1702:7
3action_distribution_network_statefulpartitionedcall
policy_cond_identityИҐ+action_distribution/StatefulPartitionedCallж
+action_distribution/StatefulPartitionedCallStatefulPartitionedCall3action_distribution_network_statefulpartitionedcallaction_distribution_1696action_distribution_1698action_distribution_1700action_distribution_1702*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *%
f R
__inference_parametrize_1695ђ
#action_distribution/PartitionedCallPartitionedCall4action_distribution/StatefulPartitionedCall:output:04action_distribution/StatefulPartitionedCall:output:14action_distribution/StatefulPartitionedCall:output:2*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *
fR
__inference_mode_1723П
policy/cond/IdentityIdentity,action_distribution/PartitionedCall:output:0^policy/cond/NoOp*
T0*#
_output_shapes
:€€€€€€€€€А
policy/cond/NoOpNoOp,^action_distribution/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "5
policy_cond_identitypolicy/cond/Identity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
: : : : :€€€€€€€€€@2Z
+action_distribution/StatefulPartitionedCall+action_distribution/StatefulPartitionedCall:-)
'
_output_shapes
:€€€€€€€€€@
©
Н
__inference_apply_1172

args_0
horizons	
deterministic

dense0_1143:	ї@
dense0_1145:@
dense1_1166:@@
dense1_1168:@
identityИҐdense0/StatefulPartitionedCallҐdense1/StatefulPartitionedCallЄ
dense0/StatefulPartitionedCallStatefulPartitionedCallargs_0dense0_1143dense0_1145*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *
fR
__inference_apply_1142ў
dense1/StatefulPartitionedCallStatefulPartitionedCall'dense0/StatefulPartitionedCall:output:0dense1_1166dense1_1168*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *
fR
__inference_apply_1165v
IdentityIdentity'dense1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@И
NoOpNoOp^dense0/StatefulPartitionedCall^dense1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:€€€€€€€€€ї:€€€€€€€€€: : : : : 2@
dense0/StatefulPartitionedCalldense0/StatefulPartitionedCall2@
dense1/StatefulPartitionedCalldense1/StatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€ї
 
_user_specified_nameargs_0:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
horizons:EA

_output_shapes
: 
'
_user_specified_namedeterministic
ъ
Ќ
"action_distribution_cond_true_1751*
&action_distribution_cond_identity_mean(
$action_distribution_cond_placeholder*
&action_distribution_cond_placeholder_1%
!action_distribution_cond_identityГ
!action_distribution/cond/IdentityIdentity&action_distribution_cond_identity_mean*
T0*#
_output_shapes
:€€€€€€€€€"O
!action_distribution_cond_identity*action_distribution/cond/Identity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :€€€€€€€€€:€€€€€€€€€: :) %
#
_output_shapes
:€€€€€€€€€:)%
#
_output_shapes
:€€€€€€€€€:

_output_shapes
: 
ґ.
†
 __inference_independent_act_1405

args_0
deterministic


agent_1387:	ї@

agent_1389:@

agent_1391:@@

agent_1393:@

agent_1395:@

agent_1397:

agent_1399:@

agent_1401:
identityИҐagent/StatefulPartitionedCallҐ agent/VerifyFinite/CheckNumericsҐ'agent/assert_equal_1/Assert/AssertGuardA
agent/ShapeShapeargs_0*
T0*
_output_shapes
:c
agent/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: e
agent/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:e
agent/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:п
agent/strided_sliceStridedSliceagent/Shape:output:0"agent/strided_slice/stack:output:0$agent/strided_slice/stack_1:output:0$agent/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask`

agent/CastCastagent/strided_slice:output:0*

DstT0	*

SrcT0*
_output_shapes
: M
agent/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 ZR
4agent/assert_type/statically_determined_correct_typeNoOp*
_output_shapes
 X
agent/Const_1Const*
_output_shapes
:*
dtype0	*
valueB	RїV
agent/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : r
agent/ExpandDims
ExpandDimsagent/Cast:y:0agent/ExpandDims/dim:output:0*
T0	*
_output_shapes
:S
agent/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Х
agent/concatConcatV2agent/ExpandDims:output:0agent/Const_1:output:0agent/concat/axis:output:0*
N*
T0	*
_output_shapes
:C
agent/Shape_1Shapeargs_0*
T0*
_output_shapes
:`
agent/Cast_1Castagent/Shape_1:output:0*

DstT0	*

SrcT0*
_output_shapes
:q
agent/assert_equal_1/EqualEqualagent/Cast_1:y:0agent/concat:output:0*
T0	*
_output_shapes
:d
agent/assert_equal_1/ConstConst*
_output_shapes
:*
dtype0*
valueB: |
agent/assert_equal_1/AllAllagent/assert_equal_1/Equal:z:0#agent/assert_equal_1/Const:output:0*
_output_shapes
: Ш
!agent/assert_equal_1/Assert/ConstConst*
_output_shapes
: *
dtype0*G
value>B< B6Agent.independent_act: invalid shape for  state input.П
#agent/assert_equal_1/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x == y did not hold element-wise:y
#agent/assert_equal_1/Assert/Const_2Const*
_output_shapes
: *
dtype0*&
valueB Bx (agent/Cast_1:0) = y
#agent/assert_equal_1/Assert/Const_3Const*
_output_shapes
: *
dtype0*&
valueB By (agent/concat:0) = «
'agent/assert_equal_1/Assert/AssertGuardIf!agent/assert_equal_1/All:output:0!agent/assert_equal_1/All:output:0agent/Cast_1:y:0agent/concat:output:0*
Tcond0
*
Tin
2
		*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *E
else_branch6R4
2agent_assert_equal_1_Assert_AssertGuard_false_1009*
output_shapes
: *D
then_branch5R3
1agent_assert_equal_1_Assert_AssertGuard_true_1008П
0agent/assert_equal_1/Assert/AssertGuard/IdentityIdentity0agent/assert_equal_1/Assert/AssertGuard:output:0*
T0
*
_output_shapes
: ю
 agent/VerifyFinite/CheckNumericsCheckNumericsargs_0(^agent/assert_equal_1/Assert/AssertGuard*
T0*
_class
loc:@args_0*(
_output_shapes
:€€€€€€€€€ї*K
message@>Agent.independent_act: invalid inf/nan value for  state input.™
%agent/VerifyFinite/control_dependencyIdentityargs_0!^agent/VerifyFinite/CheckNumerics*
T0*
_class
loc:@args_0*(
_output_shapes
:€€€€€€€€€їT
6agent/assert_type_1/statically_determined_correct_typeNoOp*
_output_shapes
 P
agent/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ≥
agent/ExpandDims_1/dimConst&^agent/VerifyFinite/control_dependency1^agent/assert_equal_1/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
value	B : v
agent/ExpandDims_1
ExpandDimsagent/Cast:y:0agent/ExpandDims_1/dim:output:0*
T0	*
_output_shapes
:Ѓ
agent/zeros/ConstConst&^agent/VerifyFinite/control_dependency1^agent/assert_equal_1/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R М
agent/zerosFillagent/ExpandDims_1:output:0agent/zeros/Const:output:0*
T0	*#
_output_shapes
:€€€€€€€€€*

index_type0	ѓ
agent/StatefulPartitionedCallStatefulPartitionedCallargs_0agent/zeros:output:0deterministic
agent_1387
agent_1389
agent_1391
agent_1393
agent_1395
agent_1397
agent_1399
agent_1401*
Tin
2	
*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:€€€€€€€€€**
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8В *"
fR
__inference_core_act_1386q
IdentityIdentity&agent/StatefulPartitionedCall:output:0^NoOp*
T0*#
_output_shapes
:€€€€€€€€€≥
NoOpNoOp^agent/StatefulPartitionedCall!^agent/VerifyFinite/CheckNumerics(^agent/assert_equal_1/Assert/AssertGuard*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€ї: : : : : : : : : 2>
agent/StatefulPartitionedCallagent/StatefulPartitionedCall2D
 agent/VerifyFinite/CheckNumerics agent/VerifyFinite/CheckNumerics2R
'agent/assert_equal_1/Assert/AssertGuard'agent/assert_equal_1/Assert/AssertGuard:P L
(
_output_shapes
:€€€€€€€€€ї
 
_user_specified_nameargs_0:EA

_output_shapes
: 
'
_user_specified_namedeterministic
ћ
с
2agent_assert_equal_1_Assert_AssertGuard_false_1526K
Gagent_assert_equal_1_assert_assertguard_assert_agent_assert_equal_1_all
@
<agent_assert_equal_1_assert_assertguard_assert_agent_maximum	@
<agent_assert_equal_1_assert_assertguard_assert_agent_const_1	6
2agent_assert_equal_1_assert_assertguard_identity_1
ИҐ.agent/assert_equal_1/Assert/AssertGuard/Assert∞
5agent/assert_equal_1/Assert/AssertGuard/Assert/data_0Const*
_output_shapes
: *
dtype0*K
valueBB@ B:Policy/baseline on-policy horizon currently not supported.°
5agent/assert_equal_1/Assert/AssertGuard/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x == y did not hold element-wise:М
5agent/assert_equal_1/Assert/AssertGuard/Assert/data_2Const*
_output_shapes
: *
dtype0*'
valueB Bx (agent/Maximum:0) = М
5agent/assert_equal_1/Assert/AssertGuard/Assert/data_4Const*
_output_shapes
: *
dtype0*'
valueB By (agent/Const_1:0) = §
.agent/assert_equal_1/Assert/AssertGuard/AssertAssertGagent_assert_equal_1_assert_assertguard_assert_agent_assert_equal_1_all>agent/assert_equal_1/Assert/AssertGuard/Assert/data_0:output:0>agent/assert_equal_1/Assert/AssertGuard/Assert/data_1:output:0>agent/assert_equal_1/Assert/AssertGuard/Assert/data_2:output:0<agent_assert_equal_1_assert_assertguard_assert_agent_maximum>agent/assert_equal_1/Assert/AssertGuard/Assert/data_4:output:0<agent_assert_equal_1_assert_assertguard_assert_agent_const_1*
T

2		*
_output_shapes
 „
0agent/assert_equal_1/Assert/AssertGuard/IdentityIdentityGagent_assert_equal_1_assert_assertguard_assert_agent_assert_equal_1_all/^agent/assert_equal_1/Assert/AssertGuard/Assert*
T0
*
_output_shapes
: …
2agent/assert_equal_1/Assert/AssertGuard/Identity_1Identity9agent/assert_equal_1/Assert/AssertGuard/Identity:output:0-^agent/assert_equal_1/Assert/AssertGuard/NoOp*
T0
*
_output_shapes
: Я
,agent/assert_equal_1/Assert/AssertGuard/NoOpNoOp/^agent/assert_equal_1/Assert/AssertGuard/Assert*"
_acd_function_control_output(*
_output_shapes
 "q
2agent_assert_equal_1_assert_assertguard_identity_1;agent/assert_equal_1/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : 2`
.agent/assert_equal_1/Assert/AssertGuard/Assert.agent/assert_equal_1/Assert/AssertGuard/Assert: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ѓ
&
__inference_value_1730
identityV
temperature/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Q
IdentityIdentitytemperature/Const:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
Й
ф
policy_cond_false_1621*
action_distribution_1732:@&
action_distribution_1734:*
action_distribution_1736:@&
action_distribution_1738:7
3action_distribution_network_statefulpartitionedcall
policy_cond_identityИҐ+action_distribution/StatefulPartitionedCallҐ-action_distribution/StatefulPartitionedCall_1с
temperature/PartitionedCallPartitionedCall*	
Tin
 *
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *
fR
__inference_value_1730ж
+action_distribution/StatefulPartitionedCallStatefulPartitionedCall3action_distribution_network_statefulpartitionedcallaction_distribution_1732action_distribution_1734action_distribution_1736action_distribution_1738*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *%
f R
__inference_parametrize_1695з
-action_distribution/StatefulPartitionedCall_1StatefulPartitionedCall4action_distribution/StatefulPartitionedCall:output:04action_distribution/StatefulPartitionedCall:output:14action_distribution/StatefulPartitionedCall:output:2$temperature/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В * 
fR
__inference_sample_1785Щ
policy/cond/IdentityIdentity6action_distribution/StatefulPartitionedCall_1:output:0^policy/cond/NoOp*
T0*#
_output_shapes
:€€€€€€€€€∞
policy/cond/NoOpNoOp,^action_distribution/StatefulPartitionedCall.^action_distribution/StatefulPartitionedCall_1*"
_acd_function_control_output(*
_output_shapes
 "5
policy_cond_identitypolicy/cond/Identity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
: : : : :€€€€€€€€€@2Z
+action_distribution/StatefulPartitionedCall+action_distribution/StatefulPartitionedCall2^
-action_distribution/StatefulPartitionedCall_1-action_distribution/StatefulPartitionedCall_1:-)
'
_output_shapes
:€€€€€€€€€@
Ъ

ё
__inference_apply_1203
x7
%linear_matmul_readvariableop_resource:@4
&linear_biasadd_readvariableop_resource:
identityИҐlinear/BiasAdd/ReadVariableOpҐlinear/MatMul/ReadVariableOpВ
linear/MatMul/ReadVariableOpReadVariableOp%linear_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0r
linear/MatMulMatMulx$linear/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€А
linear/BiasAdd/ReadVariableOpReadVariableOp&linear_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Л
linear/BiasAddBiasAddlinear/MatMul:product:0%linear/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€f
IdentityIdentitylinear/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€Е
NoOpNoOp^linear/BiasAdd/ReadVariableOp^linear/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€@: : 2>
linear/BiasAdd/ReadVariableOplinear/BiasAdd/ReadVariableOp2<
linear/MatMul/ReadVariableOplinear/MatMul/ReadVariableOp:J F
'
_output_shapes
:€€€€€€€€€@

_user_specified_namex
Ч
-
__inference_apply_1600
x
identityL
activation/TanhTanhx*
T0*'
_output_shapes
:€€€€€€€€€@[
IdentityIdentityactivation/Tanh:y:0*
T0*'
_output_shapes
:€€€€€€€€€@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:€€€€€€€€€@:J F
'
_output_shapes
:€€€€€€€€€@

_user_specified_namex
П
√
policy_cond_true_1182*
action_distribution_1263:@&
action_distribution_1265:*
action_distribution_1267:@&
action_distribution_1269:7
3action_distribution_network_statefulpartitionedcall
policy_cond_identityИҐ+action_distribution/StatefulPartitionedCallж
+action_distribution/StatefulPartitionedCallStatefulPartitionedCall3action_distribution_network_statefulpartitionedcallaction_distribution_1263action_distribution_1265action_distribution_1267action_distribution_1269*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *%
f R
__inference_parametrize_1262ђ
#action_distribution/PartitionedCallPartitionedCall4action_distribution/StatefulPartitionedCall:output:04action_distribution/StatefulPartitionedCall:output:14action_distribution/StatefulPartitionedCall:output:2*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *
fR
__inference_mode_1291П
policy/cond/IdentityIdentity,action_distribution/PartitionedCall:output:0^policy/cond/NoOp*
T0*#
_output_shapes
:€€€€€€€€€А
policy/cond/NoOpNoOp,^action_distribution/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "5
policy_cond_identitypolicy/cond/Identity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
: : : : :€€€€€€€€€@2Z
+action_distribution/StatefulPartitionedCall+action_distribution/StatefulPartitionedCall:-)
'
_output_shapes
:€€€€€€€€€@
Ж
Ѓ
1agent_assert_equal_1_Assert_AssertGuard_true_1081M
Iagent_assert_equal_1_assert_assertguard_identity_agent_assert_equal_1_all
7
3agent_assert_equal_1_assert_assertguard_placeholder	9
5agent_assert_equal_1_assert_assertguard_placeholder_1	6
2agent_assert_equal_1_assert_assertguard_identity_1
J
,agent/assert_equal_1/Assert/AssertGuard/NoOpNoOp*
_output_shapes
 „
0agent/assert_equal_1/Assert/AssertGuard/IdentityIdentityIagent_assert_equal_1_assert_assertguard_identity_agent_assert_equal_1_all-^agent/assert_equal_1/Assert/AssertGuard/NoOp*
T0
*
_output_shapes
: Ъ
2agent/assert_equal_1/Assert/AssertGuard/Identity_1Identity9agent/assert_equal_1/Assert/AssertGuard/Identity:output:0*
T0
*
_output_shapes
: "q
2agent_assert_equal_1_assert_assertguard_identity_1;agent/assert_equal_1/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : : 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
№
-
__inference_past_horizon_1065
identity	O
network/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R [
network/stackPacknetwork/Const:output:0*
N*
T0	*
_output_shapes
:_
network/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : s
network/MaxMaxnetwork/stack:output:0&network/Max/reduction_indices:output:0*
T0	*
_output_shapes
: K
IdentityIdentitynetwork/Max:output:0*
T0	*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
џ
-
__inference_past_horizon_1500
identity	ф
network/PartitionedCallPartitionedCall*	
Tin
 *
Tout
2	*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *&
f!R
__inference_past_horizon_1497W
IdentityIdentity network/PartitionedCall:output:0*
T0	*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
Ъ

ё
__inference_apply_1638
x7
%linear_matmul_readvariableop_resource:@4
&linear_biasadd_readvariableop_resource:
identityИҐlinear/BiasAdd/ReadVariableOpҐlinear/MatMul/ReadVariableOpВ
linear/MatMul/ReadVariableOpReadVariableOp%linear_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0r
linear/MatMulMatMulx$linear/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€А
linear/BiasAdd/ReadVariableOpReadVariableOp&linear_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Л
linear/BiasAddBiasAddlinear/MatMul:product:0%linear/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€f
IdentityIdentitylinear/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€Е
NoOpNoOp^linear/BiasAdd/ReadVariableOp^linear/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€@: : 2>
linear/BiasAdd/ReadVariableOplinear/BiasAdd/ReadVariableOp2<
linear/MatMul/ReadVariableOplinear/MatMul/ReadVariableOp:J F
'
_output_shapes
:€€€€€€€€€@

_user_specified_namex
ч2
Н
__inference_core_act_1816

args_0
parallel	
deterministic

policy_1796:	ї@
policy_1798:@
policy_1800:@@
policy_1802:@
policy_1804:@
policy_1806:
policy_1808:@
policy_1810:
identityИҐ'agent/assert_equal_1/Assert/AssertGuardҐpolicy/StatefulPartitionedCallP
agent/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    O
agent/Const_1Const*
_output_shapes
: *
dtype0	*
value	B	 R у
policy/PartitionedCallPartitionedCall*	
Tin
 *
Tout
2	*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *&
f!R
__inference_past_horizon_1500х
baseline/PartitionedCallPartitionedCall*	
Tin
 *
Tout
2	*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *&
f!R
__inference_past_horizon_1512}
agent/MaximumMaximumpolicy/PartitionedCall:output:0!baseline/PartitionedCall:output:0*
T0	*
_output_shapes
: o
agent/assert_equal_1/EqualEqualagent/Maximum:z:0agent/Const_1:output:0*
T0	*
_output_shapes
: [
agent/assert_equal_1/RankConst*
_output_shapes
: *
dtype0*
value	B : b
 agent/assert_equal_1/range/startConst*
_output_shapes
: *
dtype0*
value	B : b
 agent/assert_equal_1/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :Ј
agent/assert_equal_1/rangeRange)agent/assert_equal_1/range/start:output:0"agent/assert_equal_1/Rank:output:0)agent/assert_equal_1/range/delta:output:0*
_output_shapes
: |
agent/assert_equal_1/AllAllagent/assert_equal_1/Equal:z:0#agent/assert_equal_1/range:output:0*
_output_shapes
: Ь
!agent/assert_equal_1/Assert/ConstConst*
_output_shapes
: *
dtype0*K
valueBB@ B:Policy/baseline on-policy horizon currently not supported.П
#agent/assert_equal_1/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x == y did not hold element-wise:z
#agent/assert_equal_1/Assert/Const_2Const*
_output_shapes
: *
dtype0*'
valueB Bx (agent/Maximum:0) = z
#agent/assert_equal_1/Assert/Const_3Const*
_output_shapes
: *
dtype0*'
valueB By (agent/Const_1:0) = …
'agent/assert_equal_1/Assert/AssertGuardIf!agent/assert_equal_1/All:output:0!agent/assert_equal_1/All:output:0agent/Maximum:z:0agent/Const_1:output:0*
Tcond0
*
Tin
2
		*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *E
else_branch6R4
2agent_assert_equal_1_Assert_AssertGuard_false_1526*
output_shapes
: *D
then_branch5R3
1agent_assert_equal_1_Assert_AssertGuard_true_1525П
0agent/assert_equal_1/Assert/AssertGuard/IdentityIdentity0agent/assert_equal_1/Assert/AssertGuard:output:0*
T0
*
_output_shapes
: t
agent/ShapeShapeargs_01^agent/assert_equal_1/Assert/AssertGuard/Identity*
T0*
_output_shapes
:Ц
agent/strided_slice/stackConst1^agent/assert_equal_1/Assert/AssertGuard/Identity*
_output_shapes
:*
dtype0*
valueB: Ш
agent/strided_slice/stack_1Const1^agent/assert_equal_1/Assert/AssertGuard/Identity*
_output_shapes
:*
dtype0*
valueB:Ш
agent/strided_slice/stack_2Const1^agent/assert_equal_1/Assert/AssertGuard/Identity*
_output_shapes
:*
dtype0*
valueB:п
agent/strided_sliceStridedSliceagent/Shape:output:0"agent/strided_slice/stack:output:0$agent/strided_slice/stack_1:output:0$agent/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask`

agent/CastCastagent/strided_slice:output:0*

DstT0	*

SrcT0*
_output_shapes
: Ж
agent/range/startConst1^agent/assert_equal_1/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R Ж
agent/range/deltaConst1^agent/assert_equal_1/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 RН
agent/rangeRangeagent/range/start:output:0agent/Cast:y:0agent/range/delta:output:0*

Tidx0	*#
_output_shapes
:€€€€€€€€€Й
agent/ExpandDims/dimConst1^agent/assert_equal_1/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
value	B : r
agent/ExpandDims
ExpandDimsagent/Cast:y:0agent/ExpandDims/dim:output:0*
T0	*
_output_shapes
:Е
agent/ones/ConstConst1^agent/assert_equal_1/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 RИ

agent/onesFillagent/ExpandDims:output:0agent/ones/Const:output:0*
T0	*#
_output_shapes
:€€€€€€€€€*

index_type0	Е
agent/stackPackagent/range:output:0agent/ones:output:0*
N*
T0	*'
_output_shapes
:€€€€€€€€€*

axis≥
policy/StatefulPartitionedCallStatefulPartitionedCallargs_0agent/stack:output:0deterministicpolicy_1796policy_1798policy_1800policy_1802policy_1804policy_1806policy_1808policy_1810*
Tin
2	
*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:€€€€€€€€€**
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8В *
fR
__inference_act_1795t
agent/zeros_like	ZerosLike'policy/StatefulPartitionedCall:output:0*
T0*#
_output_shapes
:€€€€€€€€€
	agent/AddAddV2'policy/StatefulPartitionedCall:output:0agent/zeros_like:y:0*
T0*#
_output_shapes
:€€€€€€€€€X
IdentityIdentityagent/Add:z:0^NoOp*
T0*#
_output_shapes
:€€€€€€€€€С
NoOpNoOp(^agent/assert_equal_1/Assert/AssertGuard^policy/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:€€€€€€€€€ї:€€€€€€€€€: : : : : : : : : 2R
'agent/assert_equal_1/Assert/AssertGuard'agent/assert_equal_1/Assert/AssertGuard2@
policy/StatefulPartitionedCallpolicy/StatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€ї
 
_user_specified_nameargs_0:MI
#
_output_shapes
:€€€€€€€€€
"
_user_specified_name
parallel:EA

_output_shapes
: 
'
_user_specified_namedeterministic
Ъ

ё
__inference_apply_1664
x7
%linear_matmul_readvariableop_resource:@4
&linear_biasadd_readvariableop_resource:
identityИҐlinear/BiasAdd/ReadVariableOpҐlinear/MatMul/ReadVariableOpВ
linear/MatMul/ReadVariableOpReadVariableOp%linear_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0r
linear/MatMulMatMulx$linear/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€А
linear/BiasAdd/ReadVariableOpReadVariableOp&linear_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Л
linear/BiasAddBiasAddlinear/MatMul:product:0%linear/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€f
IdentityIdentitylinear/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€Е
NoOpNoOp^linear/BiasAdd/ReadVariableOp^linear/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€@: : 2>
linear/BiasAdd/ReadVariableOplinear/BiasAdd/ReadVariableOp2<
linear/MatMul/ReadVariableOplinear/MatMul/ReadVariableOp:J F
'
_output_shapes
:€€€€€€€€€@

_user_specified_namex
Ч
-
__inference_apply_1162
x
identityL
activation/TanhTanhx*
T0*'
_output_shapes
:€€€€€€€€€@[
IdentityIdentityactivation/Tanh:y:0*
T0*'
_output_shapes
:€€€€€€€€€@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:€€€€€€€€€@:J F
'
_output_shapes
:€€€€€€€€€@

_user_specified_namex
Ч
-
__inference_apply_1579
x
identityL
activation/TanhTanhx*
T0*'
_output_shapes
:€€€€€€€€€@[
IdentityIdentityactivation/Tanh:y:0*
T0*'
_output_shapes
:€€€€€€€€€@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:€€€€€€€€€@:J F
'
_output_shapes
:€€€€€€€€€@

_user_specified_namex
і&
и
__inference__traced_save_1892
file_prefixs
osavev2_agent_policy_action_distribution_distributions_action_distribution_mean_summary_step_read_readvariableop	P
Lsavev2_agent_policy_action_distribution_mean_linear_bias_read_readvariableopS
Osavev2_agent_policy_action_distribution_mean_linear_weights_read_readvariableopR
Nsavev2_agent_policy_action_distribution_stddev_linear_bias_read_readvariableopU
Qsavev2_agent_policy_action_distribution_stddev_linear_weights_read_readvariableop?
;savev2_agent_policy_network_dense0_bias_read_readvariableopB
>savev2_agent_policy_network_dense0_weights_read_readvariableop?
;savev2_agent_policy_network_dense1_bias_read_readvariableopB
>savev2_agent_policy_network_dense1_weights_read_readvariableop+
'savev2_save_counter_read_readvariableop	
savev2_const

identity_1ИҐMergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/partБ
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : У
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: Щ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*¬
valueЄBµBtagent.Spolicy.Saction_distribution.Sdistributions_action_distribution-mean-summary-step:0/.ATTRIBUTES/VARIABLE_VALUEBSagent.Spolicy.Saction_distribution.Smean.Slinear.Sbias:0/.ATTRIBUTES/VARIABLE_VALUEBVagent.Spolicy.Saction_distribution.Smean.Slinear.Sweights:0/.ATTRIBUTES/VARIABLE_VALUEBUagent.Spolicy.Saction_distribution.Sstddev.Slinear.Sbias:0/.ATTRIBUTES/VARIABLE_VALUEBXagent.Spolicy.Saction_distribution.Sstddev.Slinear.Sweights:0/.ATTRIBUTES/VARIABLE_VALUEBAagent.Spolicy.Snetwork.Sdense0.Sbias:0/.ATTRIBUTES/VARIABLE_VALUEBDagent.Spolicy.Snetwork.Sdense0.Sweights:0/.ATTRIBUTES/VARIABLE_VALUEBAagent.Spolicy.Snetwork.Sdense1.Sbias:0/.ATTRIBUTES/VARIABLE_VALUEBDagent.Spolicy.Snetwork.Sdense1.Sweights:0/.ATTRIBUTES/VARIABLE_VALUEB'save_counter/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHГ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*)
value BB B B B B B B B B B B Р
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0osavev2_agent_policy_action_distribution_distributions_action_distribution_mean_summary_step_read_readvariableopLsavev2_agent_policy_action_distribution_mean_linear_bias_read_readvariableopOsavev2_agent_policy_action_distribution_mean_linear_weights_read_readvariableopNsavev2_agent_policy_action_distribution_stddev_linear_bias_read_readvariableopQsavev2_agent_policy_action_distribution_stddev_linear_weights_read_readvariableop;savev2_agent_policy_network_dense0_bias_read_readvariableop>savev2_agent_policy_network_dense0_weights_read_readvariableop;savev2_agent_policy_network_dense1_bias_read_readvariableop>savev2_agent_policy_network_dense1_weights_read_readvariableop'savev2_save_counter_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2		Р
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:Л
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*\
_input_shapesK
I: : ::@::@:@:	ї@:@:@@: : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: : 

_output_shapes
::$ 

_output_shapes

:@: 

_output_shapes
::$ 

_output_shapes

:@: 

_output_shapes
:@:%!

_output_shapes
:	ї@: 

_output_shapes
:@:$	 

_output_shapes

:@@:


_output_shapes
: :

_output_shapes
: 
∆
ё
__inference_apply_1603
x7
%dense1_matmul_readvariableop_resource:@@4
&dense1_biasadd_readvariableop_resource:@
identityИҐdense1/BiasAdd/ReadVariableOpҐdense1/MatMul/ReadVariableOpВ
dense1/MatMul/ReadVariableOpReadVariableOp%dense1_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0r
dense1/MatMulMatMulx$dense1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@А
dense1/BiasAdd/ReadVariableOpReadVariableOp&dense1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Л
dense1/BiasAddBiasAdddense1/MatMul:product:0%dense1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@Э
activation/PartitionedCallPartitionedCalldense1/BiasAdd:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *
fR
__inference_apply_1600r
IdentityIdentity#activation/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@Е
NoOpNoOp^dense1/BiasAdd/ReadVariableOp^dense1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€@: : 2>
dense1/BiasAdd/ReadVariableOpdense1/BiasAdd/ReadVariableOp2<
dense1/MatMul/ReadVariableOpdense1/MatMul/ReadVariableOp:J F
'
_output_shapes
:€€€€€€€€€@

_user_specified_namex
№
-
__inference_past_horizon_1497
identity	O
network/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R [
network/stackPacknetwork/Const:output:0*
N*
T0	*
_output_shapes
:_
network/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : s
network/MaxMaxnetwork/stack:output:0&network/Max/reduction_indices:output:0*
T0	*
_output_shapes
: K
IdentityIdentitynetwork/Max:output:0*
T0	*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
c
$
"__inference_initial_internals_1838*(
_construction_contextkEagerRuntime*
_input_shapes 
∆
ё
__inference_apply_1165
x7
%dense1_matmul_readvariableop_resource:@@4
&dense1_biasadd_readvariableop_resource:@
identityИҐdense1/BiasAdd/ReadVariableOpҐdense1/MatMul/ReadVariableOpВ
dense1/MatMul/ReadVariableOpReadVariableOp%dense1_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0r
dense1/MatMulMatMulx$dense1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@А
dense1/BiasAdd/ReadVariableOpReadVariableOp&dense1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Л
dense1/BiasAddBiasAdddense1/MatMul:product:0%dense1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@Э
activation/PartitionedCallPartitionedCalldense1/BiasAdd:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *
fR
__inference_apply_1162r
IdentityIdentity#activation/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@Е
NoOpNoOp^dense1/BiasAdd/ReadVariableOp^dense1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€@: : 2>
dense1/BiasAdd/ReadVariableOpdense1/BiasAdd/ReadVariableOp2<
dense1/MatMul/ReadVariableOpdense1/MatMul/ReadVariableOp:J F
'
_output_shapes
:€€€€€€€€€@

_user_specified_namex
®
Л
__inference_apply_1671
x
linear_1665:@
linear_1667:
identityИҐlinear/StatefulPartitionedCall≥
linear/StatefulPartitionedCallStatefulPartitionedCallxlinear_1665linear_1667*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *
fR
__inference_apply_1664v
IdentityIdentity'linear/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€g
NoOpNoOp^linear/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€@: : 2@
linear/StatefulPartitionedCalllinear/StatefulPartitionedCall:J F
'
_output_shapes
:€€€€€€€€€@

_user_specified_namex
“
п
2agent_assert_equal_1_Assert_AssertGuard_false_1009K
Gagent_assert_equal_1_assert_assertguard_assert_agent_assert_equal_1_all
?
;agent_assert_equal_1_assert_assertguard_assert_agent_cast_1	?
;agent_assert_equal_1_assert_assertguard_assert_agent_concat	6
2agent_assert_equal_1_assert_assertguard_identity_1
ИҐ.agent/assert_equal_1/Assert/AssertGuard/Assertђ
5agent/assert_equal_1/Assert/AssertGuard/Assert/data_0Const*
_output_shapes
: *
dtype0*G
value>B< B6Agent.independent_act: invalid shape for  state input.°
5agent/assert_equal_1/Assert/AssertGuard/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x == y did not hold element-wise:Л
5agent/assert_equal_1/Assert/AssertGuard/Assert/data_2Const*
_output_shapes
: *
dtype0*&
valueB Bx (agent/Cast_1:0) = Л
5agent/assert_equal_1/Assert/AssertGuard/Assert/data_4Const*
_output_shapes
: *
dtype0*&
valueB By (agent/concat:0) = Ґ
.agent/assert_equal_1/Assert/AssertGuard/AssertAssertGagent_assert_equal_1_assert_assertguard_assert_agent_assert_equal_1_all>agent/assert_equal_1/Assert/AssertGuard/Assert/data_0:output:0>agent/assert_equal_1/Assert/AssertGuard/Assert/data_1:output:0>agent/assert_equal_1/Assert/AssertGuard/Assert/data_2:output:0;agent_assert_equal_1_assert_assertguard_assert_agent_cast_1>agent/assert_equal_1/Assert/AssertGuard/Assert/data_4:output:0;agent_assert_equal_1_assert_assertguard_assert_agent_concat*
T

2		*
_output_shapes
 „
0agent/assert_equal_1/Assert/AssertGuard/IdentityIdentityGagent_assert_equal_1_assert_assertguard_assert_agent_assert_equal_1_all/^agent/assert_equal_1/Assert/AssertGuard/Assert*
T0
*
_output_shapes
: …
2agent/assert_equal_1/Assert/AssertGuard/Identity_1Identity9agent/assert_equal_1/Assert/AssertGuard/Identity:output:0-^agent/assert_equal_1/Assert/AssertGuard/NoOp*
T0
*
_output_shapes
: Я
,agent/assert_equal_1/Assert/AssertGuard/NoOpNoOp/^agent/assert_equal_1/Assert/AssertGuard/Assert*"
_acd_function_control_output(*
_output_shapes
 "q
2agent_assert_equal_1_assert_assertguard_identity_1;agent/assert_equal_1/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: ::2`
.agent/assert_equal_1/Assert/AssertGuard/Assert.agent/assert_equal_1/Assert/AssertGuard/Assert: 

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:
Ъ

ё
__inference_apply_1231
x7
%linear_matmul_readvariableop_resource:@4
&linear_biasadd_readvariableop_resource:
identityИҐlinear/BiasAdd/ReadVariableOpҐlinear/MatMul/ReadVariableOpВ
linear/MatMul/ReadVariableOpReadVariableOp%linear_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0r
linear/MatMulMatMulx$linear/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€А
linear/BiasAdd/ReadVariableOpReadVariableOp&linear_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Л
linear/BiasAddBiasAddlinear/MatMul:product:0%linear/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€f
IdentityIdentitylinear/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€Е
NoOpNoOp^linear/BiasAdd/ReadVariableOp^linear/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€@: : 2>
linear/BiasAdd/ReadVariableOplinear/BiasAdd/ReadVariableOp2<
linear/MatMul/ReadVariableOplinear/MatMul/ReadVariableOp:J F
'
_output_shapes
:€€€€€€€€€@

_user_specified_namex
ч2
Н
__inference_core_act_1386

args_0
parallel	
deterministic

policy_1366:	ї@
policy_1368:@
policy_1370:@@
policy_1372:@
policy_1374:@
policy_1376:
policy_1378:@
policy_1380:
identityИҐ'agent/assert_equal_1/Assert/AssertGuardҐpolicy/StatefulPartitionedCallP
agent/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    O
agent/Const_1Const*
_output_shapes
: *
dtype0	*
value	B	 R у
policy/PartitionedCallPartitionedCall*	
Tin
 *
Tout
2	*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *&
f!R
__inference_past_horizon_1054х
baseline/PartitionedCallPartitionedCall*	
Tin
 *
Tout
2	*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *&
f!R
__inference_past_horizon_1068}
agent/MaximumMaximumpolicy/PartitionedCall:output:0!baseline/PartitionedCall:output:0*
T0	*
_output_shapes
: o
agent/assert_equal_1/EqualEqualagent/Maximum:z:0agent/Const_1:output:0*
T0	*
_output_shapes
: [
agent/assert_equal_1/RankConst*
_output_shapes
: *
dtype0*
value	B : b
 agent/assert_equal_1/range/startConst*
_output_shapes
: *
dtype0*
value	B : b
 agent/assert_equal_1/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :Ј
agent/assert_equal_1/rangeRange)agent/assert_equal_1/range/start:output:0"agent/assert_equal_1/Rank:output:0)agent/assert_equal_1/range/delta:output:0*
_output_shapes
: |
agent/assert_equal_1/AllAllagent/assert_equal_1/Equal:z:0#agent/assert_equal_1/range:output:0*
_output_shapes
: Ь
!agent/assert_equal_1/Assert/ConstConst*
_output_shapes
: *
dtype0*K
valueBB@ B:Policy/baseline on-policy horizon currently not supported.П
#agent/assert_equal_1/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x == y did not hold element-wise:z
#agent/assert_equal_1/Assert/Const_2Const*
_output_shapes
: *
dtype0*'
valueB Bx (agent/Maximum:0) = z
#agent/assert_equal_1/Assert/Const_3Const*
_output_shapes
: *
dtype0*'
valueB By (agent/Const_1:0) = …
'agent/assert_equal_1/Assert/AssertGuardIf!agent/assert_equal_1/All:output:0!agent/assert_equal_1/All:output:0agent/Maximum:z:0agent/Const_1:output:0*
Tcond0
*
Tin
2
		*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *E
else_branch6R4
2agent_assert_equal_1_Assert_AssertGuard_false_1082*
output_shapes
: *D
then_branch5R3
1agent_assert_equal_1_Assert_AssertGuard_true_1081П
0agent/assert_equal_1/Assert/AssertGuard/IdentityIdentity0agent/assert_equal_1/Assert/AssertGuard:output:0*
T0
*
_output_shapes
: t
agent/ShapeShapeargs_01^agent/assert_equal_1/Assert/AssertGuard/Identity*
T0*
_output_shapes
:Ц
agent/strided_slice/stackConst1^agent/assert_equal_1/Assert/AssertGuard/Identity*
_output_shapes
:*
dtype0*
valueB: Ш
agent/strided_slice/stack_1Const1^agent/assert_equal_1/Assert/AssertGuard/Identity*
_output_shapes
:*
dtype0*
valueB:Ш
agent/strided_slice/stack_2Const1^agent/assert_equal_1/Assert/AssertGuard/Identity*
_output_shapes
:*
dtype0*
valueB:п
agent/strided_sliceStridedSliceagent/Shape:output:0"agent/strided_slice/stack:output:0$agent/strided_slice/stack_1:output:0$agent/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask`

agent/CastCastagent/strided_slice:output:0*

DstT0	*

SrcT0*
_output_shapes
: Ж
agent/range/startConst1^agent/assert_equal_1/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R Ж
agent/range/deltaConst1^agent/assert_equal_1/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 RН
agent/rangeRangeagent/range/start:output:0agent/Cast:y:0agent/range/delta:output:0*

Tidx0	*#
_output_shapes
:€€€€€€€€€Й
agent/ExpandDims/dimConst1^agent/assert_equal_1/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
value	B : r
agent/ExpandDims
ExpandDimsagent/Cast:y:0agent/ExpandDims/dim:output:0*
T0	*
_output_shapes
:Е
agent/ones/ConstConst1^agent/assert_equal_1/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 RИ

agent/onesFillagent/ExpandDims:output:0agent/ones/Const:output:0*
T0	*#
_output_shapes
:€€€€€€€€€*

index_type0	Е
agent/stackPackagent/range:output:0agent/ones:output:0*
N*
T0	*'
_output_shapes
:€€€€€€€€€*

axis≥
policy/StatefulPartitionedCallStatefulPartitionedCallargs_0agent/stack:output:0deterministicpolicy_1366policy_1368policy_1370policy_1372policy_1374policy_1376policy_1378policy_1380*
Tin
2	
*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:€€€€€€€€€**
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8В *
fR
__inference_act_1365t
agent/zeros_like	ZerosLike'policy/StatefulPartitionedCall:output:0*
T0*#
_output_shapes
:€€€€€€€€€
	agent/AddAddV2'policy/StatefulPartitionedCall:output:0agent/zeros_like:y:0*
T0*#
_output_shapes
:€€€€€€€€€X
IdentityIdentityagent/Add:z:0^NoOp*
T0*#
_output_shapes
:€€€€€€€€€С
NoOpNoOp(^agent/assert_equal_1/Assert/AssertGuard^policy/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:€€€€€€€€€ї:€€€€€€€€€: : : : : : : : : 2R
'agent/assert_equal_1/Assert/AssertGuard'agent/assert_equal_1/Assert/AssertGuard2@
policy/StatefulPartitionedCallpolicy/StatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€ї
 
_user_specified_nameargs_0:MI
#
_output_shapes
:€€€€€€€€€
"
_user_specified_name
parallel:EA

_output_shapes
: 
'
_user_specified_namedeterministic
џ
-
__inference_past_horizon_1512
identity	ф
network/PartitionedCallPartitionedCall*	
Tin
 *
Tout
2	*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *&
f!R
__inference_past_horizon_1509W
IdentityIdentity network/PartitionedCall:output:0*
T0	*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
џ
-
__inference_past_horizon_1068
identity	ф
network/PartitionedCallPartitionedCall*	
Tin
 *
Tout
2	*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *&
f!R
__inference_past_horizon_1065W
IdentityIdentity network/PartitionedCall:output:0*
T0	*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes "њL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*ё
serving_default 
:
args_00
serving_default_args_0:0€€€€€€€€€ї
6
deterministic%
serving_default_deterministic:0
 8
output_0,
StatefulPartitionedCall:0€€€€€€€€€tensorflow/serving/predict:Р
–
ZVagent/policy/action_distribution/distributions_action_distribution-mean-summary-step:0
73agent/policy/action_distribution/mean/linear/bias:0
:6agent/policy/action_distribution/mean/linear/weights:0
95agent/policy/action_distribution/stddev/linear/bias:0
<8agent/policy/action_distribution/stddev/linear/weights:0
&"agent/policy/network/dense0/bias:0
)%agent/policy/network/dense0/weights:0
&"agent/policy/network/dense1/bias:0
)	%agent/policy/network/dense1/weights:0

save_counter
act
initial_internals

signatures"
_generic_user_object
\:Z	 2Tagent/policy/action_distribution/distributions_action_distribution-mean-summary-step
?:=21agent/policy/action_distribution/mean/linear/bias
F:D@24agent/policy/action_distribution/mean/linear/weights
A:?23agent/policy/action_distribution/stddev/linear/bias
H:F@26agent/policy/action_distribution/stddev/linear/weights
.:,@2 agent/policy/network/dense0/bias
6:4	ї@2#agent/policy/network/dense0/weights
.:,@2 agent/policy/network/dense1/bias
5:3@@2#agent/policy/network/dense1/weights
:	 2save_counter
И
trace_02л
 __inference_independent_act_1835∆
Л≤З
FullArgSpec
argsЪ 
varargsjargs
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *6Ґ3
К€€€€€€€€€ї
К
deterministic 
ztrace_0
”
trace_02ґ
"__inference_initial_internals_1838П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ ztrace_0
,
serving_default"
signature_map
ЗBД
 __inference_independent_act_1835args_0deterministic"∆
Л≤З
FullArgSpec
argsЪ 
varargsjargs
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *6Ґ3
К€€€€€€€€€ї
К
deterministic 

єBґ
"__inference_initial_internals_1838"П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ 
’B“
"__inference_signature_wrapper_1429args_0deterministic"Ф
Н≤Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 О
 __inference_independent_act_1835j	HҐE
>Ґ;
!К
args_0€€€€€€€€€ї
К
deterministic 

™ "К€€€€€€€€€Z
"__inference_initial_internals_18384Ґ

Ґ 
™ "#™ 

baseline™ 

policy™ «
"__inference_signature_wrapper_1429†	cҐ`
Ґ 
Y™V
+
args_0!К
args_0€€€€€€€€€ї
'
deterministicК
deterministic 
"/™,
*
output_0К
output_0€€€€€€€€€