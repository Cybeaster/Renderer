#include "Material.hpp"

namespace RAPI
{
SMaterialSpec OMaterial::GoldMaterial{
	{ 0.2473F, .1995F, .0745F, 1.F }, // ambient
	{ .7516F, .6065F, .2265F, 1.F }, // diffuse
	{ .6283F, .5558F, .3661F, 1.F }, // specular
	51.2F // shininess
};

SMaterialSpec OMaterial::SilverMaterial{
	{ 0.1923F, 0.1923F, 0.1923F, 1 }, // ambient
	{ 0.5075F, 0.5075F, 0.5075F, 1 }, // diffuse
	{ 0.5083F, 0.5083F, 0.5083F, 1 }, // specular
	51.2F // shininess
};
;
SMaterialSpec OMaterial::BronzeMaterial{
	{ 0.2125F, 0.1275F, 0.0540F, 1 }, // ambient
	{ 0.7140F, 0.4284F, 0.1814F, 1 }, // diffuse
	{ 0.3935F, 0.2719F, 0.1667F, 1 }, // specular
	25.16F // shininess
};
;

} // namespace RAPI
